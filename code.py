#!/usr/bin/env python3  # Shebang line to specify Python 3 interpreter
"""
Ultimate Image Text Processor with Live Camera Support
Features:
1. Extract text from image and print it
2. Edge detection on alphabets and display as words
3. Live camera OCR mode
4. Print confidence levels for all combinations
5. Validation from edge detection
"""

import cv2  # computer vision operations
import numpy as np  # NumPy for numerical array canny
import pytesseract  # Python wrapper for Tesseract OCR engine
import matplotlib.pyplot as plt  # Plotting library for displaying results
from pathlib import Path  # Modern path handling utilities
import time  # Time utilities for timestamping

class UltimateImageTextProcessor:  # Main class that handles all text processing operations
    """
         image text processing.
    """
    
    def __init__(self, tesseract_path=None):  # Constructor to initialize the processor
        if tesseract_path:  # Check if custom Tesseract path is provided
            pytesseract.pytesseract.tesseract_cmd = tesseract_path  # Set custom Tesseract executable path
    
    def smart_preprocess(self, image):  # Method to preprocess image using multiple techniques
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert color image to grayscale

        # Method 1: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(  # Apply adaptive threshold for varying lighting
            gray, 255,  # Maximum value assigned to pixel
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Use Gaussian-weighted threshold calculation
            cv2.THRESH_BINARY, 31, 2  # Binary threshold with block size 31 and constant 2
        )
        if np.mean(adaptive_thresh) > 127:  # Check if image has more white than black pixels
            adaptive_thresh = cv2.bitwise_not(adaptive_thresh)  # Invert colors to make text black on white
        denoised_adaptive = cv2.medianBlur(adaptive_thresh, 3)  # Remove noise using median filter

        # Method 2: OTSU
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply OTSU automatic thresholding

        # Method 3: Bilateral + OTSU
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)  # Apply bilateral filter to reduce noise while preserving edges
        _, bilateral_thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Apply OTSU on filtered image

        return {  # Return dictionary containing all preprocessed versions
            'adaptive': denoised_adaptive,  # Adaptive threshold result
            'otsu': otsu_thresh,  # OTSU threshold result
            'bilateral': bilateral_thresh,  # Bilateral filter + OTSU result
            'original_gray': gray  # Original grayscale image
        }
    
    def extract_text_with_confidence(self, preprocessed_images):  # Method to run OCR with multiple configs and pick best result
        print("🔍 PART 1: SMART TEXT EXTRACTION")  # Print section header
        print("=" * 45)  # Print separator line

        best_text, best_confidence, best_method = "", 0, ""  # Initialize variables to track best OCR result
        psm_modes = [6, 8, 13, 3]  # List of Page Segmentation Modes to try
        all_combinations = []  # Store all combination results

        for method_name, processed_img in preprocessed_images.items():  # Loop through each preprocessing method
            if method_name == 'original_gray':  # Skip the original grayscale image
                continue  # Move to next iteration
                
            for psm in psm_modes:  # Loop through each PSM mode
                try:  # Try OCR with current configuration
                    whitelist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()[]"\'-/\\ '  # Define allowed characters
                    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'  # Build Tesseract configuration string
                    data = pytesseract.image_to_data(processed_img, config=config, output_type=pytesseract.Output.DICT)  # Get OCR data with confidence scores
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]  # Extract positive confidence values
                    avg_conf = sum(confidences) / len(confidences) if confidences else 0  # Calculate average confidence
                    text = pytesseract.image_to_string(processed_img, config=config)  # Extract text using same config
                    
                    # Store combination result
                    combination_result = {
                        'method': method_name,
                        'psm': psm,
                        'text': text.strip(),
                        'confidence': avg_conf,
                        'char_count': len(text.strip()),
                        'word_count': len(text.strip().split()) if text.strip() else 0
                    }
                    all_combinations.append(combination_result)

                    if text.strip() and (len(text.strip()) > len(best_text.strip()) or avg_conf > best_confidence):  # Check if this result is better
                        best_text, best_confidence, best_method = text, avg_conf, f"{method_name} + PSM {psm}"  # Update best result
                        
                except Exception as e:  # If OCR fails with this configuration
                    combination_result = {
                        'method': method_name,
                        'psm': psm,
                        'text': '',
                        'confidence': 0,
                        'char_count': 0,
                        'word_count': 0,
                        'error': str(e)
                    }
                    all_combinations.append(combination_result)
                    continue  # Try next configuration

        # Print all combination results
        print("\n📊 ALL COMBINATION RESULTS:")
        print("=" * 80)
        print(f"{'Method':<12} {'PSM':<4} {'Confidence':<12} {'Words':<6} {'Chars':<6} {'Text Preview':<30}")
        print("-" * 80)
        
        for result in sorted(all_combinations, key=lambda x: x['confidence'], reverse=True):
            if 'error' in result:
                print(f"{result['method']:<12} {result['psm']:<4} {'ERROR':<12} {'-':<6} {'-':<6} {'Failed to process':<30}")
            else:
                text_preview = result['text'][:27] + "..." if len(result['text']) > 30 else result['text']
                text_preview = text_preview.replace('\n', ' ').replace('\t', ' ')
                print(f"{result['method']:<12} {result['psm']:<4} {result['confidence']:<12.1f} {result['word_count']:<6} {result['char_count']:<6} {text_preview:<30}")
        
        print("-" * 80)

        if not best_text.strip():  # If no text was found with any configuration
            try:  # Try fallback OCR method
                best_text = pytesseract.image_to_string(preprocessed_images['adaptive'], config=r'--oem 3 --psm 6')  # Use simple adaptive threshold
                best_method = "Adaptive fallback"  # Set method name
            except Exception as e:  # If even fallback fails
                best_text = f"Error: {e}"  # Set error message

        print(f"\n✅ Best method: {best_method}")  # Print the best OCR method found
        print(f"✅ Confidence: {best_confidence:.1f}%")  # Print confidence score
        print(f"✅ Text length: {len(best_text.strip())} characters")  # Print text length
        print("\n📝 Extracted Text:")  # Print text section header
        print("-" * 40)  # Print separator
        print(best_text.strip() if best_text.strip() else "❌ No text detected")  # Print extracted text or error
        print("-" * 40)  # Print separator

        return best_text.strip(), best_method, best_confidence, all_combinations  # Return best results and all combinations
    
    def extract_text_from_contours(self, image, letter_contours, words):
        """Extract text from detected letter contours using OCR on individual regions"""
        extracted_words = []
        
        for word_idx, word in enumerate(words):
            if not word:  # Skip empty words
                continue
                
            # Get bounding box for the entire word
            word_x = min([x for x, y, w, h, _ in word])
            word_y = min([y for x, y, w, h, _ in word])
            word_right = max([x + w for x, y, w, h, _ in word])
            word_bottom = max([y + h for x, y, w, h, _ in word])
            word_w = word_right - word_x
            word_h = word_bottom - word_y
            
            # Extract word region with padding
            padding = 5
            word_x = max(0, word_x - padding)
            word_y = max(0, word_y - padding)
            word_w = min(image.shape[1] - word_x, word_w + 2 * padding)
            word_h = min(image.shape[0] - word_y, word_h + 2 * padding)
            
            word_region = image[word_y:word_y + word_h, word_x:word_x + word_w]
            
            if word_region.size == 0:  # Skip if region is empty
                continue
            
            # Preprocess the word region
            if len(word_region.shape) == 3:
                word_gray = cv2.cvtColor(word_region, cv2.COLOR_BGR2GRAY)
            else:
                word_gray = word_region
                
            # Apply threshold to make text clearer
            _, word_thresh = cv2.threshold(word_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Resize if too small for better OCR
            if word_h < 20:
                scale_factor = 20 / word_h
                new_width = int(word_w * scale_factor)
                new_height = int(word_h * scale_factor)
                word_thresh = cv2.resize(word_thresh, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            try:
                # OCR on the word region
                word_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                word_text = pytesseract.image_to_string(word_thresh, config=word_config).strip()
                
                if word_text:  # Only add non-empty words
                    extracted_words.append({
                        'word': word_text,
                        'position': (word_x, word_y),
                        'letter_count': len(word),
                        'confidence': self.get_word_confidence(word_thresh, word_config)
                    })
            except Exception:
                # If OCR fails, try to estimate based on letter count
                estimated_word = '?' * len(word)
                extracted_words.append({
                    'word': estimated_word,
                    'position': (word_x, word_y),
                    'letter_count': len(word),
                    'confidence': 0
                })
        
        return extracted_words
    
    def get_word_confidence(self, word_image, config):
        """Get confidence score for a word region"""
        try:
            data = pytesseract.image_to_data(word_image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            return sum(confidences) / len(confidences) if confidences else 0
        except:
            return 0

    def detect_letter_edges_advanced(self, image, preprocessed_images):  # Method for advanced edge detection with letter grouping
        processed = preprocessed_images['adaptive']  # Use adaptive threshold result for edge detection
        blurred = cv2.GaussianBlur(processed, (3, 3), 0)  # Apply Gaussian blur to reduce noise
        edges = cv2.Canny(blurred, 50, 150)  # Apply Canny edge detection with thresholds 50 and 150
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours in the processed image
        result_image = image.copy()  # Create copy of original image for drawing

        letter_contours = []  # Initialize list to store valid letter contours
        for contour in contours:  # Loop through each found contour
            area = cv2.contourArea(contour)  # Calculate contour area
            x, y, w, h = cv2.boundingRect(contour)  # Get bounding rectangle coordinates
            if w > 5 and h > 8 and area > 25:  # Filter contours by minimum size requirements
                aspect_ratio = h / w if w > 0 else 0  # Calculate height to width ratio
                if 0.3 < aspect_ratio < 5:  # Filter by aspect ratio typical for letters
                    letter_contours.append((x, y, w, h, contour))  # Add valid contour to list

        letter_contours.sort(key=lambda x: (x[1] // 25, x[0]))  # Sort contours by row then column (reading order)
        words, current_word = [], []  # Initialize lists for words and current word being built
        for i, (x, y, w, h, contour) in enumerate(letter_contours):  # Loop through sorted letter contours
            if i > 0:  # If not the first letter
                prev_x, prev_y, prev_w, prev_h, _ = letter_contours[i-1]  # Get previous letter position
                horizontal_gap = x - (prev_x + prev_w)  # Calculate horizontal gap between letters
                vertical_gap = abs(y - prev_y)  # Calculate vertical gap between letters
                if horizontal_gap > 20 or vertical_gap > 15:  # If gap is large enough for word break
                    if current_word:  # If current word has letters
                        words.append(current_word)  # Add current word to words list
                        current_word = []  # Start new empty word
            current_word.append((x, y, w, h, contour))  # Add current letter to current word
        if current_word:  # Don't forget the last word
            words.append(current_word)  # Add final word to words list

        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]  # Define colors for different words
        for word_idx, word in enumerate(words):  # Loop through each detected word
            color = colors[word_idx % len(colors)]  # Cycle through available colors
            for letter_idx, (x, y, w, h, contour) in enumerate(word):  # Loop through letters in current word
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)  # Draw rectangle around letter
        
        # Extract text from detected contours
        edge_extracted_words = self.extract_text_from_contours(image, letter_contours, words)
        edge_extracted_text = ' '.join([word['word'] for word in edge_extracted_words])
        
        # Edge detection validation metrics
        edge_validation = {
            'total_contours': len(contours),
            'valid_letter_contours': len(letter_contours),
            'detected_words': len(words),
            'edge_density': np.sum(edges > 0) / edges.size,
            'contour_area_avg': np.mean([cv2.contourArea(contour) for _, _, _, _, contour in letter_contours]) if letter_contours else 0,
            'extracted_text': edge_extracted_text,
            'extracted_words': edge_extracted_words,
            'avg_word_confidence': np.mean([word['confidence'] for word in edge_extracted_words]) if edge_extracted_words else 0
        }
        
        return result_image, edges, processed, len(words), len(letter_contours), edge_validation  # Return results with validation
    
    def calculate_text_similarity(self, text1, text2):
        """Calculate similarity between two text strings"""
        if not text1 and not text2:
            return 1.0  # Both empty
        if not text1 or not text2:
            return 0.0  # One empty
        
        # Convert to lowercase and split into words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        jaccard = intersection / union if union > 0 else 0
        
        # Calculate character-level similarity
        from difflib import SequenceMatcher
        char_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        
        # Combine both similarities (weighted average)
        combined_similarity = (jaccard * 0.6 + char_similarity * 0.4)
        
        return combined_similarity

    def validate_ocr_with_edge_detection(self, ocr_results, edge_validation):
        """Validate OCR results using edge detection metrics and actual text comparison"""
        print("\n🔍 PART 2: EDGE DETECTION VALIDATION")
        print("=" * 45)
        
        print(f"📊 Edge Detection Metrics:")
        print(f"   • Total contours found: {edge_validation['total_contours']}")
        print(f"   • Valid letter contours: {edge_validation['valid_letter_contours']}")
        print(f"   • Detected word groups: {edge_validation['detected_words']}")
        print(f"   • Edge density: {edge_validation['edge_density']:.3f}")
        print(f"   • Average contour area: {edge_validation['contour_area_avg']:.1f}")
        print(f"   • Edge detection confidence: {edge_validation['avg_word_confidence']:.1f}%")
        
        print(f"\n📝 Edge Detection Extracted Text:")
        print(f"   '{edge_validation['extracted_text']}'")
        
        print(f"\n🔍 Individual Word Analysis:")
        for i, word_info in enumerate(edge_validation['extracted_words']):
            print(f"   Word {i+1}: '{word_info['word']}' (confidence: {word_info['confidence']:.1f}%, letters: {word_info['letter_count']})")
        
        # Validation logic with text comparison
        validation_results = []
        edge_text = edge_validation['extracted_text']
        
        for result in ocr_results:
            if 'error' in result:
                validation_score = 0
                validation_notes = "OCR processing failed"
                text_similarity = 0
            else:
                validation_score = 0
                validation_notes = []
                
                # Calculate text similarity with edge detection result
                text_similarity = self.calculate_text_similarity(result['text'], edge_text)
                similarity_points = int(text_similarity * 40)  # Up to 40 points for text similarity
                validation_score += similarity_points
                
                if text_similarity > 0.8:
                    validation_notes.append(f"Excellent text match with edge detection ({text_similarity:.2f})")
                elif text_similarity > 0.6:
                    validation_notes.append(f"Good text match with edge detection ({text_similarity:.2f})")
                elif text_similarity > 0.3:
                    validation_notes.append(f"Moderate text match with edge detection ({text_similarity:.2f})")
                else:
                    validation_notes.append(f"Poor text match with edge detection ({text_similarity:.2f})")
                
                # Check if OCR word count matches edge detection
                word_count_match = abs(result['word_count'] - edge_validation['detected_words']) <= 2
                if word_count_match:
                    validation_score += 20
                    validation_notes.append("Word count matches edge detection")
                else:
                    validation_notes.append(f"Word count mismatch (OCR: {result['word_count']}, Edge: {edge_validation['detected_words']})")
                
                # Check if there are enough contours for the detected text
                if edge_validation['valid_letter_contours'] >= result['char_count'] * 0.5:
                    validation_score += 15
                    validation_notes.append("Sufficient letter contours detected")
                else:
                    validation_notes.append("Insufficient letter contours for detected text")
                
                # Check edge density (higher density usually means more text)
                if edge_validation['edge_density'] > 0.1 and result['char_count'] > 10:
                    validation_score += 10
                    validation_notes.append("Good edge density for text detection")
                elif edge_validation['edge_density'] < 0.05 and result['char_count'] > 5:
                    validation_notes.append("Low edge density for amount of detected text")
                else:
                    validation_score += 5
                    validation_notes.append("Edge density reasonable")
                
                # Bonus for high OCR confidence
                if result['confidence'] > 70:
                    validation_score += 15
                    validation_notes.append("High OCR confidence")
                elif result['confidence'] > 50:
                    validation_score += 10
                    validation_notes.append("Medium OCR confidence")
                elif result['confidence'] > 0:
                    validation_score += 3
                    validation_notes.append("Low OCR confidence")
                
                validation_notes = "; ".join(validation_notes)
            
            validation_results.append({
                'method': result['method'],
                'psm': result['psm'],
                'ocr_text': result['text'][:50] + "..." if len(result['text']) > 50 else result['text'],
                'ocr_confidence': result['confidence'],
                'text_similarity': text_similarity,
                'validation_score': validation_score,
                'validation_notes': validation_notes,
                'combined_score': (result['confidence'] * 0.6 + validation_score * 0.4) if 'error' not in result else 0
            })
        
        # Print validation results
        print(f"\n📋 DETAILED VALIDATION RESULTS:")
        print("=" * 120)
        print(f"{'Method':<12} {'PSM':<4} {'OCR Conf':<9} {'Similarity':<10} {'Val Score':<9} {'Combined':<9} {'OCR Text Preview':<25} {'Notes':<30}")
        print("-" * 120)
        
        for val in sorted(validation_results, key=lambda x: x['combined_score'], reverse=True):
            notes_short = val['validation_notes'][:27] + "..." if len(val['validation_notes']) > 30 else val['validation_notes']
            ocr_preview = val['ocr_text'].replace('\n', ' ')[:22] + "..." if len(val['ocr_text']) > 25 else val['ocr_text'].replace('\n', ' ')
            print(f"{val['method']:<12} {val['psm']:<4} {val['ocr_confidence']:<9.1f} {val['text_similarity']:<10.2f} {val['validation_score']:<9.0f} {val['combined_score']:<9.1f} {ocr_preview:<25} {notes_short:<30}")
        
        print("-" * 120)
        
        # Text comparison summary
        print(f"\n🔤 TEXT COMPARISON SUMMARY:")
        print(f"   Edge Detection Text: '{edge_text}'")
        print(f"   Best OCR Match: '{validation_results[0]['ocr_text']}' (similarity: {validation_results[0]['text_similarity']:.2f})")
        
        # Find best validated result
        best_validated = max(validation_results, key=lambda x: x['combined_score'])
        print(f"\n🏆 BEST VALIDATED RESULT:")
        print(f"   Method: {best_validated['method']} + PSM {best_validated['psm']}")
        print(f"   OCR Confidence: {best_validated['ocr_confidence']:.1f}%")
        print(f"   Text Similarity with Edge Detection: {best_validated['text_similarity']:.2f}")
        print(f"   Validation Score: {best_validated['validation_score']:.0f}/100")
        print(f"   Combined Score: {best_validated['combined_score']:.1f}")
        
        return validation_results, best_validated
    
    def ocr_frame_realtime(self, frame):  # Method for real-time OCR processing on camera frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
        gray = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # Apply binary threshold with OTSU
        try:  # Attempt OCR processing
            data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)  # Get OCR data with bounding boxes
            extracted_text = []  # Initialize list for extracted text
            for i in range(len(data["text"])):  # Loop through each detected text element
                if int(data["conf"][i]) > 50:  # Only process text with confidence > 50%
                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]  # Get bounding box coordinates
                    text = data["text"][i]  # Get the detected text
                    if text.strip():  # If text is not empty
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw green rectangle around text
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # Display text above rectangle
                        extracted_text.append(text)  # Add text to extracted text list
            return frame, " ".join(extracted_text)  # Return processed frame and combined text
        except Exception:  # If OCR processing fails
            return frame, ""  # Return original frame with empty text
    
    def display_clean_results(self, original, preprocessed, edges, result_image, extracted_text):  # Method to display results in matplotlib grid
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))  # Create 3x2 subplot grid with specified size
        fig.subplots_adjust(hspace=0.25, wspace=0.1)  # Adjust spacing between subplots

        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # Display original image (convert BGR to RGB)
        axes[0, 0].set_title("📸 ORIGINAL IMAGE", fontsize=18, fontweight='bold', color='#34495E')  # Set title with formatting
        axes[0, 0].axis('off')  # Remove axis numbers and ticks

        axes[0, 1].imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))  # Display detection results (convert BGR to RGB)
        axes[0, 1].set_title("🎨 LETTER & WORD DETECTION", fontsize=18, fontweight='bold', color='#8E44AD')  # Set title with formatting
        axes[0, 1].axis('off')  # Remove axis numbers and ticks

        axes[1, 0].imshow(edges, cmap='gray')  # Display edge detection result in grayscale
        axes[1, 0].set_title("🔍 EDGE DETECTION", fontsize=18, fontweight='bold', color='#E67E22')  # Set title with formatting
        axes[1, 0].axis('off')  # Remove axis numbers and ticks

        axes[1, 1].imshow(preprocessed['adaptive'], cmap='gray')  # Display preprocessing result in grayscale
        axes[1, 1].set_title("🎯 PREPROCESSING", fontsize=18, fontweight='bold', color='#27AE60')  # Set title with formatting
        axes[1, 1].axis('off')  # Remove axis numbers and ticks

        axes[2, 0].axis('off')  # Turn off bottom left subplot (unused)
        
        axes[2, 1].text(  # Display extracted text in bottom right subplot
            0.01, 0.5, extracted_text,  # Position and text content
            fontsize=14, fontweight='medium', color='#2C3E50',  # Text formatting
            wrap=True, ha='left', va='center'  # Text alignment and wrapping
        )
        axes[2, 1].set_title("📝 EXTRACTED TEXT", fontsize=18, fontweight='bold', color='#2980B9')  # Set title with formatting
        axes[2, 1].axis('off')  # Remove axis numbers and ticks
        
        plt.show()  # Display the complete figure

    def process_image(self, image_path, display_results=True):
        """
        Complete processing pipeline for a single image.
        
        This is the main method that orchestrates the entire process:
        1. Load the image
        2. Preprocess it using multiple methods
        3. Extract text with confidence scoring
        4. Detect letter edges and group into words
        5. Validate OCR results with edge detection
        6. Display results (optional)
        
        Args:
            image_path (str): Path to the image file
            display_results (bool): Whether to show the results visually
        """
        # Load image using OpenCV
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Step 1: Preprocess the image
        preprocessed = self.smart_preprocess(image)
        
        # Step 2: Extract text with confidence scoring for all combinations
        extracted_text, best_method, confidence, all_combinations = self.extract_text_with_confidence(preprocessed)
        
        # Step 3: Detect letter edges and group into words
        result_image, edges, processed, word_count, letter_count, edge_validation = self.detect_letter_edges_advanced(image, preprocessed)
        
        # Step 4: Validate OCR results with edge detection
        #validation_results, best_validated = self.validate_ocr_with_edge_detection(all_combinations, edge_validation)
        
        # Step 5: Display results if requested
        if display_results:
            self.display_clean_results(image, preprocessed, edges, result_image, extracted_text)
    
    def run_live_camera(self, display_window=True):
        """
        Start live camera OCR session.
        
        This method:
        1. Opens the default camera (usually webcam)
        2. Processes each frame in real-time
        3. Displays detected text on the frame
        4. Allows user to save frames by pressing 's'
        5. Exits when user presses 'q'
        
        Args:
            display_window (bool): Whether to show the camera window
        """
        # Open camera (0 = default camera)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not access camera!")
            return
        
        # Create directory for saved captures
        Path('camera_captures').mkdir(exist_ok=True)
        capture_count = 0
        
        print("🎥 Live OCR Mode Started!")
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        
        # Main camera loop
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:  # If frame reading failed
                break
            
            # Process frame for OCR
            processed_frame, extracted_text = self.ocr_frame_realtime(frame)
            
            # Display the processed frame
            if display_window:
                cv2.imshow("🎥 Live OCR - Ultimate Processor", processed_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF  # Wait 1ms for key press
            
            if key == ord('q'):     # Quit on 'q' press
                break
            elif key == ord('s'):   # Save frame on 's' press
                filename = f"camera_captures/capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)  # Save original frame (not processed)
                capture_count += 1
                print(f"📸 Saved: {filename}")
        
        # Cleanup
        cap.release()              # Release camera
        cv2.destroyAllWindows()    # Close all windows
        print(f"✅ Camera session ended. Saved {capture_count} captures.")

def main():
    """
    Main function that provides user interface for the application.
    
    Allows user to choose between:
    1. Processing a single image file
    2. Running live camera OCR
    """
    # Configuration - UPDATE THESE PATHS FOR YOUR SYSTEM
    IMAGE_PATH = r"C:\Users\partt\Pictures\Screenshots\Screenshot 2025-06-24 215150.png"
    TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows path
    
    # Display main menu
    print("🚀 IMAGE TEXT PROCESSOR")
    print("=" * 50)
    print("1️⃣ Process single image")
    print("2️⃣ Live camera OCR")
    
    # Get user choice
    mode = input("Enter choice (1 or 2): ").strip()
    
    # Initialize processor with Tesseract path
    processor = UltimateImageTextProcessor(tesseract_path=TESSERACT_PATH)
    
    # Execute based on user choice
    if mode == "1":
        print(f"\n🖼️ Processing image: {IMAGE_PATH}")
        processor.process_image(IMAGE_PATH)
    elif mode == "2":
        print("\n🎥 Starting live camera mode...")
        processor.run_live_camera()
    else:
        print("❌ Invalid choice! Please enter 1 or 2.")

# Entry point - only run if script is executed directly (not imported)
if __name__ == "__main__":
    main()
