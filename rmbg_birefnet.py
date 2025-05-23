import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

class BackgroundRemover:
    def __init__(self, 
                 model_name='briaai/RMBG-2.0', 
                 image_size=(512, 512), 
                 device='cuda'):
        """
        Initialize the background removal model.
        
        Args:
            model_name (str): Hugging Face model identifier
            image_size (tuple): Desired input image size
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        # Validate device
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available. Falling back to CPU.")
            device = 'cpu'
        
        self.device = device
        self.image_size = image_size
        
        # Model and preprocessing setup
        self.model = AutoModelForImageSegmentation.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            trust_remote_code=True
        )
        
        # Set model precision and move to device
        torch.set_float32_matmul_precision('high')
        self.model.to(self.device)
        self.model.eval()
        
        # Image transformation pipeline
        self.transform_image = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __call__(self, input_image):
        """
        Remove background from an input image.
        
        Args:
            input_image (PIL.Image): Input image
        
        Returns:
            PIL.Image: Image with transparent background
        """
        # Preprocess image
        input_tensor = self.transform_image(input_image).unsqueeze(0).to(self.device)
        
        # Perform background removal
        with torch.no_grad():
            with torch.autocast(self.device, dtype=torch.float16 if self.device == 'cuda' else torch.float32):
                preds = self.model(input_tensor)[-1].sigmoid().cpu()
        
        # Process prediction mask
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(input_image.size)
        
        # Create image with transparent background
        result_image = input_image.copy()
        result_image.putalpha(mask)
        
        return result_image

