
def calculate_image_porosity(img_tensor):
    """
    Calculate porosity from an image tensor
    
    Args:
        img_tensor: Tensor of shape [C, H, W] with values in [-1, 1]
        
    Returns:
        Porosity value (scalar) between 0 and 1
    """
    # Convert to [0, 1] range
    img = (img_tensor + 1) / 2
    
    # Convert to grayscale if RGB
    if img.size(0) == 3:
        # Use luminance formula
        img_gray = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    else:
        img_gray = img.squeeze(0)
    
    # Apply threshold to create binary image (pores are assumed to be dark)
    # Adjust threshold if needed for your specific microstructure images
    threshold = 0.5
    binary = (img_gray < threshold).float()
    
    # Calculate porosity as ratio of pore pixels to total pixels
    porosity = binary.sum() / (binary.size(0) * binary.size(1))
    
    return porosity


