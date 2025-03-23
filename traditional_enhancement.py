import cv2
import numpy as np

class ImprovedEnhancement:
    def __init__(self, 
                 use_clahe=True, 
                 use_bilateral=True, 
                 use_gamma=True,
                 use_unsharp=True,
                 use_dehazing=True,
                 use_contrast_stretch=True,
                 use_adaptive_gamma=True,
                 clahe_clip=3.0,
                 clahe_grid=(8, 8),
                 bilateral_d=7,
                 bilateral_sigma_color=30,
                 bilateral_sigma_space=30,
                 gamma=1.5,
                 unsharp_strength=0.7,
                 dehazing_strength=0.6,
                 night_mode=True):
        """
        Initialize enhanced image processing pipeline optimized for low-light conditions
        
        Args:
            use_clahe: Use CLAHE for contrast enhancement
            use_bilateral: Use bilateral filtering for noise reduction
            use_gamma: Use gamma correction for brightness adjustment
            use_unsharp: Use unsharp masking for edge enhancement
            use_dehazing: Use dehazing to improve visibility
            use_contrast_stretch: Use adaptive contrast stretching
            use_adaptive_gamma: Use adaptive gamma based on image brightness
            clahe_clip: Clipping limit for CLAHE
            clahe_grid: Grid size for CLAHE
            bilateral_d: Diameter of each pixel neighborhood
            bilateral_sigma_color: Filter sigma in the color space
            bilateral_sigma_space: Filter sigma in the coordinate space
            gamma: Base gamma correction value
            unsharp_strength: Strength of unsharp masking
            dehazing_strength: Strength of dehazing effect
            night_mode: Enable specific optimizations for nighttime scenes
        """
        self.use_clahe = use_clahe
        self.use_bilateral = use_bilateral
        self.use_gamma = use_gamma
        self.use_unsharp = use_unsharp
        self.use_dehazing = use_dehazing
        self.use_contrast_stretch = use_contrast_stretch
        self.use_adaptive_gamma = use_adaptive_gamma
        self.clahe_clip = clahe_clip
        self.clahe_grid = clahe_grid
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.gamma = gamma
        self.unsharp_strength = unsharp_strength
        self.dehazing_strength = dehazing_strength
        self.night_mode = night_mode
        
        # Initialize CLAHE
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
    
    def apply_clahe(self, image):
        """Apply CLAHE for contrast enhancement"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # Split channels
        l, a, b = cv2.split(lab)
        # Apply CLAHE to L channel
        l = self.clahe.apply(l)
        # Merge channels
        lab = cv2.merge((l, a, b))
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def apply_bilateral_filter(self, image):
        """Apply bilateral filtering for noise reduction while preserving edges"""
        # For night scenes, use more aggressive noise reduction
        d = self.bilateral_d
        sigma_color = self.bilateral_sigma_color
        sigma_space = self.bilateral_sigma_space
        
        if self.night_mode:
            # Apply a lighter filter first for extreme noise
            pre_filtered = cv2.fastNlMeansDenoisingColored(
                image, None, 5, 5, 7, 15
            )
            # Then apply bilateral for edge preservation
            filtered = cv2.bilateralFilter(
                pre_filtered, d, sigma_color, sigma_space
            )
        else:
            filtered = cv2.bilateralFilter(
                image, d, sigma_color, sigma_space
            )
            
        return filtered
    
    def apply_gamma_correction(self, image, custom_gamma=None):
        """Apply gamma correction for brightness adjustment"""
        if custom_gamma is None:
            custom_gamma = self.gamma
            
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / custom_gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
        ]).astype(np.uint8)
        
        # Apply gamma correction using lookup table
        return cv2.LUT(image, table)
    
    def apply_adaptive_gamma(self, image):
        """Apply adaptive gamma correction based on image brightness"""
        # Convert to grayscale and calculate average brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray) / 255.0
        
        # Adjust gamma based on brightness
        if avg_brightness < 0.3:  # Very dark image
            adaptive_gamma = self.gamma + 0.7  # Higher gamma (brightens more)
        elif avg_brightness < 0.5:  # Moderately dark
            adaptive_gamma = self.gamma + 0.3
        else:  # Adequately bright
            adaptive_gamma = self.gamma
            
        return self.apply_gamma_correction(image, adaptive_gamma)
    
    def apply_unsharp_masking(self, image):
        """Apply unsharp masking for edge enhancement"""
        # Create blurred version of image
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # For night scenes, use a more controlled unsharp mask
        if self.night_mode:
            # Convert to LAB to apply unsharp only to L channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            blurred_l = cv2.GaussianBlur(l, (0, 0), 3)
            sharpened_l = cv2.addWeighted(
                l, 1.0 + self.unsharp_strength,
                blurred_l, -self.unsharp_strength,
                0
            )
            
            # Merge back
            result = cv2.merge([sharpened_l, a, b])
            return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        else:
            # Standard unsharp mask
            return cv2.addWeighted(
                image, 1.0 + self.unsharp_strength,
                blurred, -self.unsharp_strength,
                0
            )
    
    def apply_contrast_stretch(self, image):
        """Apply adaptive contrast stretching"""
        # Apply to each channel separately for better color preservation
        channels = cv2.split(image)
        result_channels = []
        
        for ch in channels:
            # Get 1st and 99th percentiles for robust stretching
            p1 = np.percentile(ch, 1)
            p99 = np.percentile(ch, 99)
            
            # Stretch contrast
            stretched = np.clip((ch - p1) * 255.0 / (p99 - p1), 0, 255).astype(np.uint8)
            result_channels.append(stretched)
        
        return cv2.merge(result_channels)
    
    def apply_dehazing(self, image):
        """Apply improved dehazing suited for nighttime scenes with lights"""
        # Convert to float
        img_float = image.astype(np.float32) / 255.0
        
        # Get dark channel
        min_channel = np.min(img_float, axis=2)
        kernel_size = 15
        dark_channel = cv2.erode(min_channel, np.ones((kernel_size, kernel_size)))
        
        # Estimate atmospheric light (A)
        flat_dark = dark_channel.flatten()
        size = flat_dark.size
        num_brightest = max(int(size * 0.001), 1)
        indices = np.argpartition(flat_dark, -num_brightest)[-num_brightest:]
        
        # Get the brightest pixels in the original image
        atm_light = np.zeros(3, dtype=np.float32)
        for ch in range(3):
            channel_flat = img_float[:,:,ch].flatten()
            atm_light[ch] = np.mean(np.percentile(channel_flat[indices], 90))
            
        # Limit atmospheric light for nighttime scenes to prevent over-brightening
        if self.night_mode:
            atm_light = np.clip(atm_light, 0.5, 0.9)
        
        # Estimate transmission map (modified for nighttime)
        omega = self.dehazing_strength  # Strength of dehazing
        transmission = 1 - omega * dark_channel / (np.max(atm_light) + 1e-6)
        
        # Refine transmission map
        refined_t = cv2.GaussianBlur(transmission, (kernel_size, kernel_size), 0)
        
        # Apply transmission map to recover scene radiance
        refined_t = np.clip(refined_t, 0.2, 1.0)  # Higher min value for nighttime
        
        # Apply to each channel
        result = np.zeros_like(img_float)
        for ch in range(3):
            t = np.expand_dims(refined_t, 2).repeat(3, axis=2)
            result = (img_float - atm_light) / t + atm_light
            
        # Scaling and normalization
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # For night mode, blend with original to preserve lights
        if self.night_mode:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v = hsv[:,:,2]
            # Find bright regions (lights)
            bright_mask = (v > 200).astype(np.float32)
            # Smooth mask
            bright_mask = cv2.GaussianBlur(bright_mask, (15, 15), 0)
            
            # Blend dehazed with original (keep original in bright areas)
            blend_factor = np.expand_dims(bright_mask, 2).repeat(3, axis=2)
            blended = image * blend_factor + result * (1 - blend_factor)
            return blended.astype(np.uint8)
        
        return result
    
    def apply_white_balance(self, image):
        """Apply automatic white balance using gray world assumption"""
        # Calculate the per-channel means
        b_avg, g_avg, r_avg = np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])
        
        # Calculate the average of the averages
        avg = (b_avg + g_avg + r_avg) / 3
        
        # Calculate scaling factors
        b_scale, g_scale, r_scale = avg / b_avg, avg / g_avg, avg / r_avg
        
        # Apply the scaling to balance the image
        balanced = image.copy().astype(np.float32)
        balanced[:,:,0] = np.clip(balanced[:,:,0] * b_scale, 0, 255)
        balanced[:,:,1] = np.clip(balanced[:,:,1] * g_scale, 0, 255)
        balanced[:,:,2] = np.clip(balanced[:,:,2] * r_scale, 0, 255)
        
        return balanced.astype(np.uint8)
    
    def enhance_night_detection(self, image):
        """Special processing for nighttime object detection"""
        # Apply a combination of techniques for night scenes
        
        # 1. Initial noise reduction
        denoised = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 15)
        
        # 2. Increase brightness and contrast
        hsv = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.equalizeHist(hsv[:,:,2])
        brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 3. Apply mild sharpening to enhance edges
        sharpened = self.apply_unsharp_masking(brightened)
        
        # 4. Apply light dehazing
        dehazed = self.apply_dehazing(sharpened)
        
        # 5. Final CLAHE for contrast
        result = self.apply_clahe(dehazed)
        
        return result
    
    def process(self, image):
        """
        Process an image through the enhanced pipeline
        
        Args:
            image (numpy.ndarray): Input image in BGR format
            
        Returns:
            numpy.ndarray: Enhanced image
        """
        # Special night detection mode
        if self.night_mode:
            # For nighttime scenes with cars, people, etc.
            return self.enhance_night_detection(image)
        
        # Standard enhancement pipeline
        enhanced = image.copy()
        
        # Apply white balance first
        enhanced = self.apply_white_balance(enhanced)
        
        # Apply contrast stretch if enabled
        if self.use_contrast_stretch:
            enhanced = self.apply_contrast_stretch(enhanced)
        
        # Apply CLAHE if enabled
        if self.use_clahe:
            enhanced = self.apply_clahe(enhanced)
        
        # Apply dehazing if enabled (before gamma correction)
        if self.use_dehazing:
            enhanced = self.apply_dehazing(enhanced)
        
        # Apply bilateral filter if enabled
        if self.use_bilateral:
            enhanced = self.apply_bilateral_filter(enhanced)
        
        # Apply gamma correction if enabled
        if self.use_gamma:
            if self.use_adaptive_gamma:
                enhanced = self.apply_adaptive_gamma(enhanced)
            else:
                enhanced = self.apply_gamma_correction(enhanced)
        
        # Apply unsharp masking if enabled
        if self.use_unsharp:
            enhanced = self.apply_unsharp_masking(enhanced)
        
        return enhanced

# Update the main RealTimeDetector code to use this improved enhancement
def update_detector_enhancement():
    """
    Function to update RealTimeDetector to use the improved enhancement pipeline
    Import this function and call it after creating the detector instance
    """
    enhancer = ImprovedEnhancement(
        use_clahe=True,
        use_bilateral=True,
        use_gamma=True,
        use_unsharp=True,
        use_dehazing=True,
        use_contrast_stretch=True,
        use_adaptive_gamma=True,
        clahe_clip=3.0,
        gamma=1.8,
        unsharp_strength=0.5,
        dehazing_strength=0.6,
        night_mode=True
    )
    
    return enhancer

# Usage example
if __name__ == "__main__":
    # Create enhancement pipeline
    enhancer = ImprovedEnhancement(
        night_mode=True,
        gamma=1.8,
        dehazing_strength=0.6
    )
    
    # Load test image
    image = cv2.imread("foggy_image.jpg")
    
    # Process with improved methods
    enhanced = enhancer.process(image)
    
    # Save result
    cv2.imwrite("enhanced_improved.jpg", enhanced)
    
    # Display before/after
    combined = np.hstack((image, enhanced))
    cv2.imshow("Before | After", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()