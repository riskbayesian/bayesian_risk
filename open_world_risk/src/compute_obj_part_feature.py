# Standard library imports
import os
import random
import time
from typing import Any, Generator, Iterable, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError

# Third-party imports
import cv2
import maskclip_onnx
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoTokenizer, CLIPTextModelWithProjection

# Local imports
from .video_datastructures import FeatureExtractionResults

def show_annotations(anns):
    if len(anns) == 0:
        return
    img = np.ones((anns.shape[1], anns.shape[2], 4))
    img[:,:,3] = 0
    for ann in range(anns.shape[0]):
        m = anns[ann].bool()
        m=m.cpu().numpy()
        color_mask = np.concatenate([np.random.random(3), [1]])
        img[m] = color_mask
    return img

def batch_iterator(batch_size: int, *args) -> Generator[List[Any], None, None]:
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def resize_image_np(image: np.ndarray, mode: str = "image", long_edge_size: int = 1024, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Efficiently resize a single image (H, W, C) or a batch of images (N, H, W, C) so that the longest edge is sam_size or shape, using PyTorch for batch mode.
    Args:
        image: np.ndarray, shape (H, W, C) or (N, H, W, C)
        shape: Optional (H, W) to resize to. If None, will resize so the largest dimension is sam_size.
        mode: "image" or "mask", changes interpolation method so that we don't lose information when resizing masks
    Returns:
        np.ndarray: Resized image(s), same number of dimensions as input.
    """
    def resize_single(img, target_shape, mode):
        if mode == "image":
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
        elif mode == "mask":
            return cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError("Invalid mode. Must be 'image' or 'mask'.")

    # Determine target shape
    if shape is not None:
        target_h, target_w, _ = shape
    else:
        img0 = image[0] if image.ndim == 4 else image
        h, w = img0.shape[:2]
        if h > w:
            target_h = long_edge_size
            target_w = int(long_edge_size * w / h)
        else:
            target_w = long_edge_size
            target_h = int(long_edge_size * h / w)
    target_shape = (target_h, target_w)

    if image.ndim == 4:
        # Batch of images: (N, H, W, C)
        images_torch = torch.from_numpy(image).permute(0, 3, 1, 2).float() / 255.0  # (N, C, H, W)
        if mode == "image":
            resized = F.interpolate(images_torch, size=target_shape, mode='bilinear', align_corners=False)
        elif mode == "mask":
            resized = F.interpolate(images_torch, size=target_shape, mode='nearest')
        else:
            raise ValueError("Invalid mode. Must be 'image' or 'mask'.")
        resized = (resized.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (N, H, W, C)
        return resized
    elif image.ndim == 3 or image.ndim == 2:
        # Single image
        return resize_single(image, target_shape, mode)
    else:
        raise ValueError("Input must be (H, W, C) or (N, H, W, C)")
    
def scale_bounding_boxes(boxes: torch.Tensor, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> torch.Tensor:
    """
    Scale bounding box coordinates from old image size to new image size.
    
    Args:
        boxes: torch.Tensor of shape (n, 4) with coordinates [x1, y1, x2, y2]
        old_size: Tuple of (height, width) of original image
        new_size: Tuple of (height, width) of new image
        
    Returns:
        torch.Tensor: Scaled bounding boxes with same shape as input
    """
    old_h, old_w = old_size
    new_h, new_w = new_size
    
    # Calculate scaling factors
    scale_h = new_h / old_h
    scale_w = new_w / old_w
    
    # Create scaling tensor
    scale_tensor = torch.tensor([scale_w, scale_h, scale_w, scale_h], device=boxes.device, dtype=boxes.dtype)
    
    # Scale the boxes
    scaled_boxes = boxes * scale_tensor
    
    return scaled_boxes

def resize_image_torch(image: torch.Tensor, mode: str = "image", long_edge_size: int = 1024, shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """
    Efficiently resize a single image (H, W, C) or a batch of images (N, H, W, C) or masks (N, H, W) so that the longest edge is sam_size or shape, using PyTorch for batch mode.
    Args:
        image: torch.Tensor, shape (H, W, C), (N, H, W, C), or (N, H, W)
        shape: Optional (H, W) to resize to. If None, will resize so the largest dimension is sam_size.
        mode: "image" or "mask", changes interpolation method so that we don't lose information when resizing masks
    Returns:
        torch.Tensor: Resized image(s), same number of dimensions as input.
    """
    if shape is not None:
        target_h, target_w = shape[:2]  # Handle both (H, W) and (H, W, C) shapes
    else:
        img0 = image[0] if image.ndim == 4 else image
        h, w = img0.shape[:2]
        if h > w:
            target_h = long_edge_size
            target_w = int(long_edge_size * w / h)
        else:
            target_w = long_edge_size
            target_h = int(long_edge_size * h / w)
    target_shape = (target_h, target_w)
    
    needs_squeeze, squeeze_dim = False, None
    # Handle different input tensor shapes
    if image.ndim == 3 and image.shape[0] == 3:  # (C, H, W) - single image
        # Add batch dimension
        image = image.unsqueeze(0)  # (1, C, H, W)
        needs_squeeze, squeeze_dim = True, 0
    elif image.ndim == 3 and image.shape[0] != 3:  # (N, H, W) - batch of masks
        # Add channel dimension
        image = image.unsqueeze(1)  # (N, 1, H, W)
        needs_squeeze, squeeze_dim = True, 1
    elif image.ndim == 4:  # (N, C, H, W) - batch of images
        needs_squeeze, squeeze_dim = False, None
    else:
        raise ValueError(f"Unsupported tensor shape: {image.shape}. Expected (C, H, W), (N, H, W), or (N, C, H, W)")
    
    if mode == "image":
        resized = F.interpolate(image, size=target_shape, mode='bilinear', align_corners=False)
    elif mode == "mask":
        resized = F.interpolate(image, size=target_shape, mode='nearest')
    else:
        raise ValueError("Invalid mode. Must be 'image' or 'mask'.")
    
    # Remove the added dimension if it was added
    if needs_squeeze:
        resized = torch.squeeze(resized, squeeze_dim)
    
    return resized

# maskclip_onnx is a library that allows us to use the CLIP model to extract features from images
# it is a wrapper around the CLIP model that allows us to use it in a more convenient way
# it is a PyTorch module that can be used in a similar way to a regular PyTorch model
class MaskCLIPFeaturizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.preprocess = maskclip_onnx.clip.load(
            "ViT-L/14@336px",
            download_root=os.getenv('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
        )
        self.model.eval()
        self.patch_size = self.model.visual.patch_size

    def forward(self, img):
        b, _, input_size_h, input_size_w = img.shape
        patch_h = input_size_h // self.patch_size
        patch_w = input_size_w // self.patch_size
        features = self.model.get_patch_encodings(img).to(torch.float32)
        return features.reshape(b, patch_h, patch_w, -1).permute(0, 3, 1, 2)

class TextCLIPFeaturizer:
    """
    Lightweight wrapper around Hugging Face CLIP text encoder with projection.

    - Loads a CLIP text model (e.g., ViT-L/14(-336)) and tokenizer
    - Provides batched `.encode(...)` that returns L2-normalized embeddings
    - Ensures outputs live in the same shared space as CLIP image embeddings

    Example
    -------
    >>> enc = textCLIPFeaturizer("openai/clip-vit-large-patch14-336")
    >>> vecs = enc.encode(["a photo of a cat", "a photo of a dog"])  # [2, 768]
    >>> sims = vecs @ vecs.T  # cosine similarity since normalized
    """

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14-336",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer & model with projection head using retry logic
        self.tokenizer, self.model = self.load_clip_text_model_with_retry()
        self.model = self.model.eval().to(self.device)

    @torch.inference_mode()
    def encode(
        self,
        texts: Union[str, Iterable[str]],
        *,
        batch_size: int = 32,
        normalize: bool = True,
        return_tensors: str = "pt",  # "pt" or "np"
        truncation: bool = True,
        padding: Union[bool, str] = True,
        max_length: Optional[int] = None,
    ):
        """Tokenize and encode text(s) into CLIP's shared embedding space.

        Parameters
        ----------
        texts : str | Iterable[str]
            Single string or iterable of strings.
        batch_size : int
            Batch size for encoding.
        normalize : bool
            L2-normalize embeddings (recommended for cosine similarity).
        return_tensors : {"pt", "np"}
            Return PyTorch tensor or NumPy array.
        truncation : bool
            Whether to truncate sequences to max length.
        padding : bool | str
            Padding strategy for tokenizer (True/"longest"/"max_length").
        max_length : Optional[int]
            If provided, pad/truncate to this length (CLIP commonly uses 77).
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)

        out_chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encode_kwargs = dict(
                padding=("max_length" if max_length is not None else padding),
                truncation=truncation,
                return_tensors="pt",
            )
            if max_length is not None:
                encode_kwargs["max_length"] = max_length

            inputs = self.tokenizer(batch, **encode_kwargs).to(self.device)
            outputs = self.model(**inputs)
            embeds = outputs.text_embeds  # [B, D] e.g., D=768 for ViT-L/14
            if normalize:
                embeds = torch.nn.functional.normalize(embeds, dim=-1)
            out_chunks.append(embeds)

        result = torch.cat(out_chunks, dim=0) if out_chunks else torch.empty((0, 0))
        if return_tensors == "np":
            return result.detach().cpu().numpy()
        return result

    def to(self, device: str):
        """Move model to a different device and return self (fluent API)."""
        self.device = device
        self.model.to(device)
        return self

    @property
    def dim(self) -> Optional[int]:
        """Projected embedding dimensionality (e.g., 768 for ViT-L/14)."""
        # Hugging Face sets this in config for CLIP text w/ projection
        return getattr(self.model.config, "projection_dim", None)

    def load_clip_text_model_with_retry(self, max_retries=5, initial_delay=5):
        """Attempt to load a CLIP text model with exponential backoff retry logic.
        
        Args:
            model_id: Hugging Face model ID (e.g., "openai/clip-vit-large-patch14-336")
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds (default: 5)
            
        Returns:
            Tuple of (tokenizer, model)
            
        Raises:
            RuntimeError: If all retry attempts fail
        """
        model_id = self.model_id
        delay = initial_delay
        last_error = None
        
        for attempt in range(max_retries):
            try:
                print(f"Loading CLIP text model {model_id} (attempt {attempt + 1}/{max_retries})...")
                # Prefer safetensors to avoid torch.load vulnerability path
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    use_fast=True
                )
                model = CLIPTextModelWithProjection.from_pretrained(
                    model_id,
                    use_safetensors=True,
                    low_cpu_mem_usage=True,
                    torch_dtype="auto"
                )
                print(f"CLIP text model {model_id} loaded successfully")
                return tokenizer, model
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Add jitter to prevent thundering herd (10-20% of delay)
                    jitter = random.uniform(0.1 * delay, 0.2 * delay)
                    sleep_time = delay + jitter
                    print(f"Failed to load CLIP text model (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {sleep_time:.1f} seconds... Error: {str(e)}")
                    time.sleep(sleep_time)
                    # Exponential backoff (2x)
                    delay *= 2
                else:
                    print(f"All {max_retries} attempts to load the CLIP text model failed.")
                    raise RuntimeError(f"Failed to load CLIP text model {model_id} after {max_retries} attempts. Last error: {str(last_error)}")

def load_model_with_retry(repo, model_name, max_retries=5, initial_delay=10):
    """Attempt to load a model with exponential backoff retry logic.
    
    Args:
        repo: Repository name (e.g., "RogerQi/MobileSAMV2")
        model_name: Name of the model to load
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds (default: 10)
        
    Returns:
        The loaded model components
        
    Raises:
        RuntimeError: If all retry attempts fail
    """
    delay = initial_delay
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"Loading MobileSAMV2 models (attempt {attempt + 1}/{max_retries})...")
            models = torch.hub.load(repo, model_name)
            print("MobileSAMV2 models loaded successfully")
            return models
            
        except (HTTPError, URLError) as e:
            last_error = e
            if attempt < max_retries - 1:
                # Add jitter to prevent thundering herd (10-20% of delay)
                jitter = random.uniform(0.1 * delay, 0.2 * delay)
                sleep_time = delay + jitter
                print(f"Failed to load model (attempt {attempt + 1}/{max_retries}). "
                      f"Retrying in {sleep_time:.1f} seconds... Error: {str(e)}")
                time.sleep(sleep_time)
                # More aggressive exponential backoff (3x instead of 2x)
                delay *= 3
            else:
                print(f"All {max_retries} attempts to load the model failed.")
                raise RuntimeError(f"Failed to load model after {max_retries} attempts. Last error: {str(last_error)}")

class FeatureExtractor:
    def __init__(self, sam_size=1024, obj_feat_res=100, final_feat_res=64, mobilesamv2_encoder_name="mobilesamv2_efficientvit_l2"):
        """Initialize the feature extractor with all necessary components.
        
        Args:
            args: Configuration arguments containing:
                - sam_size: Size for SAM processing
                - obj_feat_res: Resolution for object features
                - final_feat_res: Final feature resolution
                - mobilesamv2_encoder_name: Name of the SAM encoder
        """
        # Setup device and parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_conf = 0.4
        self.yolo_iou = 0.9
        self.sam_size = sam_size
        self.obj_feat_res = obj_feat_res
        self.final_feat_res = final_feat_res
        self.mobilesamv2_encoder_name = mobilesamv2_encoder_name
        
        # Setup models and transforms
        self.transforms = self.setup_transforms()
        self.models = self.setup_models()
    
    def setup_transforms(self):
        """Setup all necessary transforms for different models."""
        norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        raw_transform = T.Compose([
            T.ToTensor(),
            norm
        ])
        
        return {
            'raw_transform': raw_transform,
        }
    # See the following for more details:
    # https://github.com/RogerQi/MobileSAMV2/blob/main/hubconf.py
    # This gives us the models used for mobilesamev2, objawaremodel, and predictor
    # Sam Predictor: https://github.com/RogerQi/MobileSAMV2/blob/main/mobilesamv2/predictor.py
    #   Predict masks for the given input prompts, using the currently set image
    def setup_models(self):
        """Initialize and setup all required models."""
        clip_model = MaskCLIPFeaturizer().cuda().eval()
        # TextCLIPFeaturizer is not a torch.nn.Module; move internal model to device via its to() helper
        text_clip_model = TextCLIPFeaturizer().to(self.device)
        
        # Load MobileSAMV2 models with retry logic
        mobilesamv2, ObjAwareModel, predictor = load_model_with_retry(
            "RogerQi/MobileSAMV2",
            self.mobilesamv2_encoder_name,
            max_retries=5,     # Will try up to 5 times
            initial_delay=10   # Start with 10 second delay
        )
        mobilesamv2.to(device=self.device)
        mobilesamv2.eval()
        
        return {
            'clip_model': clip_model,
            'mobilesamv2': mobilesamv2,
            'ObjAwareModel': ObjAwareModel,
            'predictor': predictor,
            'text_clip_model': text_clip_model
        }

    def process_sam_masks(self, image):
        """Process SAM masks for object detection."""
        timings = {}
        
        t0 = time.time()
        obj_results = self.models['ObjAwareModel'](image, device=self.device, imgsz=self.sam_size, 
                                                 conf=self.yolo_conf, iou=self.yolo_iou, verbose=False)
        timings['yolo_detection'] = time.time() - t0
        
        t0 = time.time()
        self.models['predictor'].set_image(image)
        input_boxes1 = obj_results[0].boxes.xyxy
        input_boxes = input_boxes1.cpu().numpy()
        input_boxes = self.models['predictor'].transform.apply_boxes(input_boxes, self.models['predictor'].original_size)
        input_boxes = torch.from_numpy(input_boxes).cuda().detach()
        timings['box_prep'] = time.time() - t0
        
        # Calculate optimal batch size based on available memory
        batch_size = min(320, input_boxes.shape[0])
        while batch_size > 0:
            try:
                t0 = time.time()
                sam_mask = []
                image_embedding = self.models['predictor'].features
                image_embedding = torch.repeat_interleave(image_embedding, batch_size, dim=0)
                prompt_embedding = self.models['mobilesamv2'].prompt_encoder.get_dense_pe()
                prompt_embedding = torch.repeat_interleave(prompt_embedding, batch_size, dim=0)
                timings['embedding_prep'] = time.time() - t0
                
                t0 = time.time()
                for (boxes,) in batch_iterator(batch_size, input_boxes):
                    with torch.no_grad():
                        image_embedding = image_embedding[0:boxes.shape[0],:,:,:]
                        prompt_embedding = prompt_embedding[0:boxes.shape[0],:,:,:]
                        sparse_embeddings, dense_embeddings = self.models['mobilesamv2'].prompt_encoder(
                            points=None,
                            boxes=boxes,
                            masks=None,)
                        low_res_masks, _ = self.models['mobilesamv2'].mask_decoder(
                            image_embeddings=image_embedding,
                            image_pe=prompt_embedding,
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                            simple_type=True,
                        )
                        low_res_masks = self.models['predictor'].model.postprocess_masks(low_res_masks, 
                                                                                       self.models['predictor'].input_size, 
                                                                                       self.models['predictor'].original_size)
                        sam_mask_pre = (low_res_masks > self.models['mobilesamv2'].mask_threshold)*1.0
                        sam_mask.append(sam_mask_pre.squeeze(1))
                timings['mask_generation'] = time.time() - t0
                
                return torch.cat(sam_mask), input_boxes1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    batch_size = batch_size // 2
                    if batch_size == 0:
                        raise RuntimeError("Failed to process masks even with batch size 1")
                    continue
                else:
                    raise e
    
    def process_object_level_features_mask(self, image, sam_masks, verbose=False):
        """Process object-level CLIP features and return a list of [mask, clip_embedding] pairs."""
        timings = {}
        
        t0 = time.time()
        raw_input_image = self.transforms['raw_transform'](Image.fromarray(image))
        whole_image_feature = self.models['clip_model'](raw_input_image[None].cuda().detach())[0]
        timings['pixelwise_clip'] = time.time() - t0
        
        t0 = time.time()
        resized_clip_feat_map_bchw = torch.nn.functional.interpolate(
            whole_image_feature.unsqueeze(0).float(),
            size=(self.obj_feat_res, self.obj_feat_res),
            mode='bilinear',
            align_corners=False
        )
        # resized_clip_feat_map_bchw: [1, C, R, R], C=1024 for ViT-L/14
        timings['feature_resize'] = time.time() - t0
        
        t0 = time.time()
        masks_torch = sam_masks.float()
        
        # Preallocate tensors on GPU
        masks_tensor_list = []
        embeddings_tensor_list = []
        valid_mask_indices = []
        timings['mask_prep'] = time.time() - t0

        batch_size = 32
        for i in range(0, len(masks_torch), batch_size):
            batch_masks = masks_torch[i:i + batch_size]
            batch_masks_tensor = batch_masks.to(self.device).detach()
            resized_masks = torch.nn.functional.interpolate(
                batch_masks_tensor.unsqueeze(1),
                size=(self.obj_feat_res, self.obj_feat_res),
                mode='nearest'
            ).squeeze(1).bool()
            
            for j, resized_mask in enumerate(resized_masks):
                if resized_mask.sum() == 0:
                    continue
                mask_clip_feat = resized_clip_feat_map_bchw[0, :, resized_mask]  # [C, N]
                mask_avg_feat = mask_clip_feat.mean(dim=1).detach()              # [C]

                mask_embed_proj = mask_avg_feat

                masks_tensor_list.append(batch_masks[j])
                embeddings_tensor_list.append(mask_embed_proj)
                valid_mask_indices.append(i + j)  # Track the original index

        masks_tensor = torch.stack(masks_tensor_list)
        embeddings_tensor = torch.stack(embeddings_tensor_list)
        valid_mask_indices = torch.tensor(valid_mask_indices, device=self.device)         
        
        # GUARANTEE: Normalize each CLIP embedding to have L2 norm = 1
        # This ensures proper cosine similarity calculations
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, dim=1, p=2)
        
        # Print detailed timings for this function
        if verbose:
            print("\nFeature extraction timings:")
            for step, t in timings.items():
                print(f"{step:25s}: {t:.3f}s")
        
        return masks_tensor, embeddings_tensor, valid_mask_indices
    
    def save_visualizations(self, image, sam_mask, input_boxes1=None, obj_feat_path=None, return_visualizations=False):
        """Save various visualizations for debugging and analysis."""
        results = {}

        # Create SAM mask visualization
        annotation = sam_mask
        areas = torch.sum(annotation, dim=(1, 2))
        sorted_indices = torch.argsort(areas, descending=True)
        show_img = annotation[sorted_indices]
        ann_img = show_annotations(show_img)
        mask_visualization = ann_img

        if obj_feat_path:
            save_img_path = obj_feat_path.replace('.npy', '_mask.png')
            Image.fromarray((ann_img * 255).astype(np.uint8)).save(save_img_path)

        # Create bbox visualization
        if input_boxes1 is None:
            # Return blank image and empty bboxes in correct format
            bbox_visualization = np.zeros_like(image)
            bboxes = np.zeros((0, 4), dtype=int)
            bboxes_for_save = []
        else:
            viz_img = image.copy()
            bboxes_for_save = []
            for bbox_idx in range(input_boxes1.shape[0]):
                bbox = input_boxes1[bbox_idx]
                bbox_xyxy = bbox.cpu().numpy().astype(int)
                bboxes_for_save.append(bbox_xyxy)
                cv2.rectangle(viz_img, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), (0, 255, 0), 2)
                cv2.putText(viz_img, f'{bbox_idx}', (bbox_xyxy[0], bbox_xyxy[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            bbox_visualization = viz_img
            bboxes = np.stack(bboxes_for_save) if bboxes_for_save else np.zeros((0, 4), dtype=int)

            if obj_feat_path:
                save_img_path = obj_feat_path.replace('.npy', '_bbox.png')
                Image.fromarray(viz_img).save(save_img_path)

                # Save bounding boxes as .npy
                bbox_save_path = obj_feat_path.replace('.npy', '_bboxes.npy')
                np.save(bbox_save_path, bboxes)

        if return_visualizations:
            return {
                'mask_visualization': mask_visualization,
                'bbox_visualization': bbox_visualization,
                'bboxes': bboxes
            }

    def process_single_image(self, image, verbose=False):
        """Core function to process a single image through all feature extraction pipelines.
        This is the main processing function that both process_single_image and process_single_image_from_array use.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            output_paths: Optional dictionary of output paths
            save: Whether to save outputs
            
        Returns:
            FeatureExtractionResults: Results containing extracted features and visualizations
        """
        timings = {}
        time0 = time.time()

        # Step 1: Image preprocessing
        t0 = time.time()
        original_shape = image.shape
        image = resize_image_np(image, mode="image", long_edge_size=self.sam_size)
        processed_shape = image.shape  # Capture the actual processed image size
        timings['image_preprocessing'] = time.time() - t0

        # Step 2: SAM predictor setup and box preparation
        t0 = time.time()
        sam_masks_torch, input_boxes1 = self.process_sam_masks(image)
        timings['sam_masks_&_yolo'] = time.time() - t0
        
        # Step 3: CLIP feature extraction
        t0 = time.time()
        sam_masks_tensor, embeddings_tensor, valid_mask_indices = self.process_object_level_features_mask(image, sam_masks_torch)
        timings['mask_features_&_clip'] = time.time() - t0
        
        # Filter input_boxes to match the masks that survived empty mask filtering
        input_boxes1 = input_boxes1[valid_mask_indices]
        
        # Step 4: Output resizing
        t0 = time.time()
        image = resize_image_np(image, mode="image", shape=original_shape)
        sam_masks_tensor = resize_image_torch(sam_masks_tensor, mode="mask", shape=(original_shape[0], original_shape[1]))
        timings['output_resizing'] = time.time() - t0

        # Step 5: Final data preparation
        t0 = time.time()
        timings['sam_masks_clip'] = time.time() - t0

        # Step 6: Bounding box scaling
        t0 = time.time()
        processed_size = (processed_shape[0], processed_shape[1])
        original_size = (original_shape[0], original_shape[1])
        input_boxes1 = scale_bounding_boxes(input_boxes1, processed_size, original_size)
        timings['bbox_scaling'] = time.time() - t0

        timings['total_time'] = time.time() - time0
        
        # Print detailed timings
        if verbose:
            print("\nDetailed timings:")
            for step, t in timings.items():
                print(f"{step:25s}: {t:.3f}s")
        
        # Create and return the Pydantic model
        return FeatureExtractionResults(
            image=image,
            sam_masks=sam_masks_tensor,
            input_boxes=input_boxes1,
            clip_embeddings=embeddings_tensor,
            timings=timings
        )

    def process_list_text(self, text_lst: List[str], batch_size: int = 64):
        """
        Process a list of text strings through the CLIP text model to get embeddings.
        
        Args:
            text_lst: List of text strings to process
            batch_size: Batch size for processing (default: 64)
            
        Returns:
            torch.Tensor: CLIP text embeddings of shape [num_texts, embedding_dim]
        """
        if not text_lst:
            return torch.empty((0, self.models["text_clip_model"].dim), device=self.text_clip_model.device)
        
        # Process text through the CLIP text model
        embeddings = self.models["text_clip_model"].encode(
            texts=text_lst,
            batch_size=batch_size,
            normalize=True,  # L2 normalize for cosine similarity
            return_tensors="pt"  # Return PyTorch tensor
        )
        
        # Move to CPU to match stored frame clip embeddings (CPU in pipeline)
        embeddings = embeddings.detach().cpu()
        
        return embeddings
