"""
Pie chart processor module for analyzing pie chart images.

This module provides functionality to analyze pie chart images and extract
segment data using OpenCV and color detection algorithms.
"""


import cv2
import numpy as np
from typing import Dict
from datetime import datetime
from sklearn.cluster import KMeans


SEGMENT_COLORS = np.array(
    [
        [166, 197, 231],
        [100, 117, 186],
        [60, 112, 176],
        [150, 85, 101],
        [221, 166, 174],
        [215, 192, 154],
        [238, 238, 212],
        [138, 198, 181],
        [227, 234, 241],
    ],
    dtype=int,
)


def show_image(title, image, scale=0.7):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(title, resized)
    cv2.waitKey(0)
    # cv2.waitKey(100)


def preprocess_image(img):
    return img

    # Convert to LAB color space and back to normalize colors
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])

    # Convert back to BGR
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return normalized


def detect_color_contours(img, target_bgr, sensitivity=20, debug_mode=False):

    # 2. Define BGR Range directly (no HSV conversion)
    if debug_mode:
        print(f"Target BGR: B={target_bgr[0]}, G={target_bgr[1]}, R={target_bgr[2]}")
        print(f"Sensitivity: {sensitivity}")

    # 3. Define BGR Range
    # All channels are 0-255
    lower_bound = np.array([
        max(0, target_bgr[0] - sensitivity), 
        max(0, target_bgr[1] - sensitivity), 
        max(0, target_bgr[2] - sensitivity)
    ], dtype=np.uint8)
    upper_bound = np.array([
        min(255, target_bgr[0] + sensitivity),
        min(255, target_bgr[1] + sensitivity), 
        min(255, target_bgr[2] + sensitivity)
    ], dtype=np.uint8)
    
    if debug_mode:
        print(f"Lower bound: {lower_bound}")
        print(f"Upper bound: {upper_bound}")

    # 4. Create Mask and Clean Up (using BGR image directly)
    mask = cv2.inRange(img, lower_bound, upper_bound)
    
    # Remove small noise and bridge small gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 5. Find and Draw Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug_mode:
        print(f"Found {len(contours)} contours")
    
    if debug_mode:
        output_img = img.copy()
        cv2.drawContours(output_img, contours, -1, (0, 255, 0), 2) # Draw in Green

        # Show results
        cv2.imshow("Original", img)
        cv2.imshow("Mask", mask)
        cv2.imshow("Detected Contours", output_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return contours



def is_curved_contour(contour, min_complexity=10):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < min_complexity:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 100
    if aspect_ratio > 5:  # likely text or line
        return False

    return True


def detect_pie_colors(img, n_colors=9, debug_mode=False):
    img_processed = preprocess_image(img)

    pixels = img_processed.reshape(-1, 3).astype(np.float32)

    # Calculate grayscale brightness for each pixel
    # Standard formula: 0.299*R + 0.587*G + 0.114*B
    brightness = np.dot(pixels, [0.299, 0.587, 0.114])

    # Only discard pixels that are truly near-white (e.g., > 252)
    # and keep the lower bound to ignore black text
    mask = (brightness < 252) & (brightness > 15)

    #mask = np.all(pixels < 250, axis=1) & np.all(pixels > 10, axis=1)
    filtered_pixels = pixels[mask]
    
    if debug_mode:
        filtered_img = np.zeros_like(pixels)
        filtered_img[mask] = filtered_pixels
        filtered_img = filtered_img.reshape(img_processed.shape)
        show_image("Filtered Pixels (for color detection)", filtered_img.astype(np.uint8))
        
    if len(filtered_pixels) < 100:
        if debug_mode:
            print("Not enough colored pixels found for color detection")
        return None

    if debug_mode:
        print(f"Analyzing {len(filtered_pixels)} colored pixels...")

    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(filtered_pixels)
    detected_colors = kmeans.cluster_centers_.astype(int)

    labels = kmeans.labels_
    color_counts = np.bincount(labels)
    sorted_indices = np.argsort(-color_counts)
    detected_colors = detected_colors[sorted_indices]

    if debug_mode:
        print(f"Detected {len(detected_colors)} colors:")
        color_bar = np.zeros((100, n_colors * 50, 3), dtype=np.uint8)
        for i, color in enumerate(detected_colors):
            color_bar[:, i * 50 : (i + 1) * 50] = color
        show_image("Detected Colors", color_bar)

    return detected_colors


def reorder_colors_to_standard(
    detected_colors, max_distance_warn=25, max_distance_error=50, debug_mode=False
):
    detected = np.array(detected_colors, dtype=int)
    standard = SEGMENT_COLORS.astype(int)

    dists = np.linalg.norm(detected[:, None, :] - standard[None, :, :], axis=2)

    used = set()
    reordered = []

    for std_idx in range(standard.shape[0]):
        cand_order = np.argsort(dists[:, std_idx])
        chosen = next((c for c in cand_order if c not in used), cand_order[0])
        used.add(chosen)
        dist = dists[chosen, std_idx]

        if debug_mode:
            print(
                f"Standard #{std_idx+1}: standard={standard[std_idx]} -> detected={detected[chosen]} (dist={dist:.1f})"
            )
        if dist > max_distance_warn:
            if debug_mode:
                print(
                    f"Warning: large distance for standard #{std_idx+1} (>{max_distance_warn})"
                )
        if dist > max_distance_error:
            # Always print errors
            print(
                f"Error: distance for standard #{std_idx+1} exceeds error threshold (>{max_distance_error})"
            )
            return None

        reordered.append(detected[chosen])

    return np.array(reordered, dtype=int)


def has_long_straight_segments(cnt, img_shape, max_line_ratio=0.3):
    if cnt is None or len(cnt) < 2:
        return False

    # 1. Define the maximum allowed length based on image dimensions
    img_h, img_w = img_shape[:2]
    max_allowed_len = max(img_h, img_w) * max_line_ratio
    
    # 2. Approximate the contour to simplify curves into straight lines
    # Epsilon is the maximum distance between the original curve and its approximation
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
    
    # 3. Iterate through segments
    for i in range(len(approx)):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % len(approx)][0] # Connects back to start for closed loop
        
        # Calculate Euclidean distance between points
        line_len = np.sqrt(np.sum((p1 - p2) ** 2))
        
        if line_len > max_allowed_len:
            return True # Found a line that is too long
            
    return False


def convert_colors_to_hsv_ranges(bgr_colors, hue_tolerance=10, sat_tolerance=50, val_tolerance=50, debug_mode=False):
    """
    Convert BGR colors to HSV color ranges for use with cv2.inRange().
    
    Args:
        bgr_colors: numpy array of BGR colors [[B,G,R], ...]
        hue_tolerance: tolerance for Hue channel (0-180)
        sat_tolerance: tolerance for Saturation channel (0-255)
        val_tolerance: tolerance for Value channel (0-255)
        debug_mode: Enable debug output
        
    Returns:
        List of tuples [(lower_hsv, upper_hsv), ...] for each color
    """
    print(f"\n🎨 Converting {len(bgr_colors)} BGR colors to HSV ranges")
    print("=" * 70)
    
    hsv_ranges = []
    
    for i, bgr_color in enumerate(bgr_colors):
        # Create a 1x1 pixel image with the color
        pixel = np.uint8([[bgr_color]])
        
        # Convert to HSV
        hsv_pixel = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
        
        # Create ranges with tolerance
        # For Hue, handle wrapping around (0-180 in OpenCV)
        h_lower = max(0, h - hue_tolerance)
        h_upper = min(180, h + hue_tolerance)
        
        s_lower = max(0, s - sat_tolerance)
        s_upper = min(255, s + sat_tolerance)
        
        v_lower = max(0, v - val_tolerance)
        v_upper = min(255, v + val_tolerance)
        
        lower = (h_lower, s_lower, v_lower)
        upper = (h_upper, s_upper, v_upper)
        
        hsv_ranges.append((lower, upper))
        
        if debug_mode:
            print(f"\nSegment {i+1}:")
            print(f"  BGR: {list(bgr_color)}")
            print(f"  HSV: ({h}, {s}, {v})")
            print(f"  Range: {lower} to {upper}")
    
    print("\n" + "="*70)
    print("COPY THIS TO USE IN YOUR CODE:")
    print("="*70 + "\n")
    print("hsv_ranges = [")
    for i, (lower, upper) in enumerate(hsv_ranges):
        print(f"    ({lower}, {upper}),  # Segment {i+1}")
    print("]\n")
    print("="*70 + "\n")
    
    return hsv_ranges


def crop_pie_chart(img, debug_mode=False):
    if debug_mode:
        show_image("Original Image", img)

    img_processed = preprocess_image(img)

    if debug_mode:
        show_image("Preprocessed Image", img_processed)

    gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    thresh2 = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_and(thresh, thresh2)

    if debug_mode:
        show_image("Combined Mask", combined_mask)

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3
    )

    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    if debug_mode:
        show_image("Cleaned Mask", combined_mask)

    contours, hierarchy = cv2.findContours(
        combined_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # Find innermost contours (contours with no children)
    curved_contours = []
    if hierarchy is not None:
        hierarchy = hierarchy[0]  # Remove extra dimension
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            # Check if this contour has no children (is innermost) and meets size criteria
            has_children = hierarchy[i][2] != -1  # [2] is first_child index
            has_long_straight_lines = has_long_straight_segments(cnt, img.shape, max_line_ratio=0.5)
            if not has_long_straight_lines and area > 10000 and is_curved_contour(cnt) and not has_children:
                curved_contours.append(cnt)

    if debug_mode and len(contours) > 0:
        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 0, 255), 2)
        cv2.drawContours(debug_img, curved_contours, -1, (0, 255, 0), 3)
        show_image("Filtered Contours (Green=Curved, Red=Rejected)", debug_img)        

    if len(curved_contours) > 0:
        largest_curved_contour = max(curved_contours, key=cv2.contourArea)

        if debug_mode:
            debug_img = img.copy()
            cv2.drawContours(debug_img, [largest_curved_contour], -1, (0, 255, 0), 3)
            show_image("Largest Curved Contour (Pie Chart)", debug_img)

        (cx, cy), radius = cv2.minEnclosingCircle(largest_curved_contour)
        cx, cy, radius = int(cx), int(cy), int(radius)

        if debug_mode:
            img_with_circle = img.copy()
            cv2.circle(img_with_circle, (cx, cy), radius, (255, 0, 0), 3)
            show_image("Detected Circle", img_with_circle)

        padding = 1
        x = max(0, cx - radius - padding)
        y = max(0, cy - radius - padding)
        w = min(img.shape[1] - x, 2 * (radius + padding))
        h = min(img.shape[0] - y, 2 * (radius + padding))

        cropped = img[y : y + h, x : x + w]

        if debug_mode:
            print(f"Pie chart detected and cropped")
            print(
                f"Using largest curved contour (area: {cv2.contourArea(largest_curved_contour):.0f} pixels)"
            )
            print(
                f"Filtered out {len(curved_contours) - 1} smaller curved and {len(contours) - len(curved_contours)} straight-line contours"
            )
            print(f"Circle: center=({cx}, {cy}), radius={radius}")
            print(f"Crop region: x={x}, y={y}, width={w}, height={h}")

        return cropped
    else:
        if debug_mode:
            print("No pie chart detected!")
        return None


def measure_pie_chart(img, segment_colors, debug_mode=False):
    total_area = 0
    color_areas = []
    result_img = img.copy()

    img_processed = preprocess_image(img)

    for i, color in enumerate(segment_colors):
        tolerance = 15
        lower = np.clip(color - tolerance, 0, 255)
        upper = np.clip(color + tolerance, 0, 255)

        mask = cv2.inRange(img_processed, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 500:
                clean_mask = np.zeros_like(mask)
                cv2.drawContours(clean_mask, [largest_contour], -1, 255, -1)

                color_areas.append(
                    {
                        "name": f"{i+1}",
                        "color": color,
                        "area": area,
                        "mask": clean_mask,
                        "contour": largest_contour,
                    }
                )
                total_area += area

                # cv2.drawContours(result_img, [largest_contour], -1, tuple(map(int, color)), 3)

                if debug_mode:
                    segment_img = img.copy()
                    segment_img[clean_mask == 0] = [255, 255, 255]
                    show_image(f"Segment {i+1}", segment_img)

    expected_segments = len(segment_colors)
    if len(color_areas) != expected_segments:
        # Always print errors
        print(f"ERROR: Expected {expected_segments} segments, but only detected {len(color_areas)}")
        return None

    if debug_mode:
        print("\n" + "=" * 50)
        print("PIE CHART ANALYSIS RESULTS")
        print("=" * 50)
        print(f"All {expected_segments} segments detected successfully")
        print("-" * 50)
    
    for color_data in color_areas:
        percentage = (color_data["area"] / total_area) * 100
        color_data["percentage"] = percentage
        if debug_mode:
            print(f"{color_data['name']}: {percentage:.2f}%")
    
    if debug_mode:
        print("=" * 50)

    if debug_mode:
        height, width = img.shape[:2]
        legend_width = 300
        legend_img = np.ones((height, width + legend_width, 3), dtype=np.uint8) * 255
        legend_img[:, :width] = result_img

        y_offset = 30
        line_height = 40
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2

        cv2.putText(
            legend_img, "Legend:", (width + 10, y_offset), font, 0.8, (0, 0, 0), 2
        )
        y_offset += line_height

        for color_data in color_areas:
            color_box_size = 25
            cv2.rectangle(
                legend_img,
                (width + 10, y_offset - 20),
                (width + 10 + color_box_size, y_offset - 20 + color_box_size),
                tuple(map(int, color_data["color"])),
                -1,
            )
            cv2.rectangle(
                legend_img,
                (width + 10, y_offset - 20),
                (width + 10 + color_box_size, y_offset - 20 + color_box_size),
                (0, 0, 0),
                1,
            )

            text = f"{color_data['percentage']:.1f}%"
            cv2.putText(
                legend_img,
                text,
                (width + 45, y_offset),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )
            y_offset += line_height

        show_image("Final Result", legend_img)

    return color_areas


def is_pie_segment(cnt, min_area=500):
    # 1. Filter by size to remove noise
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False

    # 2. Approximate the shape
    # Pie segments typically have 3 core points (center and two arc ends)
    # plus a few points to describe the curve of the arc.
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    # 3. Shape Analysis: Triangle-like or sector-like
    # Should have a small number of vertices after approximation
    if not (3 <= len(approx) <= 6):
        return False

    # 4. Check Convexity and Solidity
    # Pie segments are highly convex. 
    # Solidity = Area / Convex Hull Area. Should be close to 1.
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    if solidity < 0.90: # High solidity means no deep "dents"
        return False

    # 5. Triangle Similarity (Check area ratio vs minimum enclosing triangle)
    # A pie slice is roughly a triangle.
    try:
        _, triangle = cv2.minEnclosingTriangle(cnt)
        if triangle is not None:
            tri_area = cv2.contourArea(triangle)
            # A sector of a circle has area close to its enclosing triangle
            # (Area of sector / Area of enclosing triangle) is typically 0.6 - 0.9
            if tri_area > 0:
                tri_ratio = area / tri_area
                if 0.7 < tri_ratio < 0.95:
                    return True
    except:
        # Fallback if minEnclosingTriangle fails
        pass


    # 6. Extent check (Area / Bounding Box Area)
    # For a triangle, extent is 0.5. For a sector, it's usually 0.4 - 0.7
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    if 0.3 < extent < 0.8:
        return True

    return False


def analyze_pie_chart_with_contours(
    img: np.ndarray, image_id: str, image_name: str, debug: bool = False
) -> Dict:
    """
    Analyze pie chart and return segment data.

    Args:
        img: Input image as numpy array
        image_id: Unique identifier for the image
        image_name: Name of the image
        debug: Enable debug mode for verbose output

    Returns:
        Dictionary with analysis results including:
        - success: Boolean indicating if analysis succeeded
        - id: Image identifier
        - name: Image name
        - date: Timestamp of analysis
        - segment_1 through segment_9: Percentage values for each segment (if successful)
    """
    if debug:
        print(f"[PieChart] Analyzing image: {image_name}")
    
    cropped_pie = crop_pie_chart(img, debug_mode=debug)

    default_result = {
        "success": False,
        "id": image_id,
        "name": image_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if cropped_pie is None:
        if debug:
            print(f"[PieChart] Failed to crop pie chart from image: {image_name}")
        return default_result

    segment_colors = detect_pie_colors(cropped_pie, n_colors=9, debug_mode=debug)

    if segment_colors is None:
        if debug:
            print(f"[PieChart] Failed to detect colors in image: {image_name}")
        return default_result

    segment_colors = reorder_colors_to_standard(segment_colors, max_distance_warn=25, max_distance_error=50, debug_mode=debug)
    if segment_colors is None:
        if debug:
            print(f"[PieChart] Failed to match colors to standard in image: {image_name}")
        return default_result

    color_data = measure_pie_chart(cropped_pie, segment_colors, debug_mode=debug)

    if color_data is None:
        if debug:
            print(f"[PieChart] Failed to measure pie chart segments in image: {image_name}")
        return default_result

    result = {
        "success": True,
        "id": image_id,
        "name": image_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    for i, data in enumerate(color_data, start=1):
        result[f"segment_{i}"] = round(data["percentage"], 2)

    if debug:
        print(f"[PieChart] Successfully analyzed image: {image_name}")
        for i, data in enumerate(color_data, start=1):
            print(f"[PieChart]   Segment {i}: {data['percentage']:.2f}%")

    return result


def extract_color_from_single_image(image_path):
    """
    Extract the dominant color from a single-color reference image.
    
    Args:
        image_path: Path to the single-color reference image
        
    Returns:
        numpy array with BGR color values [B, G, R], or None if extraction fails
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
       
    # Get all pixels and compute the median color (robust to noise)
    pixels = img.reshape(-1, 3).astype(np.float32)
    median_color = np.median(pixels, axis=0).astype(int)
    
    return median_color


def extract_segment_colors_from_images(image_paths, debug_mode=False):
    """
    Extract SEGMENT_COLORS from individual reference images (one per segment).
    Each image should contain only one solid color representing that segment.
    
    Args:
        image_paths: List of image paths in order (segment 1, segment 2, ...)
        debug_mode: Enable debug output and visual displays
        
    Returns:
        numpy array of colors in BGR format
        
    Example:
        image_paths = [
            "colors/segment1.png",  # Color for segment 1
            "colors/segment2.png",  # Color for segment 2
            ...
        ]
    """
    print(f"\n🎨 Extracting segment colors from {len(image_paths)} reference images")
    print("=" * 70)
    
    extracted_colors = []
    
    for i, img_path in enumerate(image_paths):        
        color = extract_color_from_single_image(img_path)
        
        if color is None:
            if debug_mode:
                print(f"  ❌ Error: Could not load image")
            return None
                
        extracted_colors.append(color)
    
    colors_array = np.array(extracted_colors, dtype=int)
        
    if debug_mode:
        print("\n" + "="*70)
        print("COPY THIS TO REPLACE SEGMENT_COLORS IN api.py:")
        print("="*70 + "\n")
        print("SEGMENT_COLORS = np.array([")
        for i, color in enumerate(colors_array):
            print(f"    {list(color)},  # Segment {i+1}")
        print("], dtype=int)\n")
        print("="*70 + "\n")
        
        # Show color bar with all extracted colors
        color_bar = np.zeros((100, len(colors_array) * 50, 3), dtype=np.uint8)
        for i, color in enumerate(colors_array):
            color_bar[:, i * 50 : (i + 1) * 50] = color
        #show_image("Extracted Segment Colors (in order)", color_bar)
    
    return colors_array


def calculate_clustering_score(contours):
    """Calculate how tightly clustered the contours are (lower = better)."""
    if len(contours) < 2:
        return float('inf')
    
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centroids.append((cx, cy))
    
    # Calculate variance of distances from mean center
    mean_x = sum(c[0] for c in centroids) / len(centroids)
    mean_y = sum(c[1] for c in centroids) / len(centroids)
    
    variance = sum(
        (cx - mean_x)**2 + (cy - mean_y)**2
        for cx, cy in centroids
    ) / len(centroids)
    
    return variance


def build_segment_set(start_color_idx, color_candidates, start_candidate):
    """Build a complete set of segments starting from a specific candidate."""
    selected = [start_candidate]
    indices = [start_color_idx]
    
    for i, candidates in enumerate(color_candidates):
        if i == start_color_idx or not candidates:
            continue
        
        # Calculate centroids of already selected segments
        centroids = []
        for contour in selected:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        
        # Find closest candidate
        best_candidate = None
        min_avg_distance = float('inf')
        
        for candidate in candidates:
            M = cv2.moments(candidate)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                avg_distance = sum(
                    np.sqrt((cx - ex)**2 + (cy - ey)**2)
                    for ex, ey in centroids
                ) / len(centroids)
                
                if avg_distance < min_avg_distance:
                    min_avg_distance = avg_distance
                    best_candidate = candidate
        
        if best_candidate is not None:
            selected.append(best_candidate)
            indices.append(i)
    
    return selected, indices

def load_color_scheme(scheme_num, debug_mode=False):
    """Load color scheme from a numbered folder structure."""
    segment_images = [
        f"colors/scheme_{scheme_num}/segment_{i}.png"
        for i in range(1, 10)
    ]
    colors = extract_segment_colors_from_images(segment_images, debug_mode=debug_mode)
    return colors if colors is not None else None


def analyze_pie_chart_with_colors(
    img: np.ndarray, 
    image_id: str, 
    image_name: str, 
    debug: bool = False
) -> Dict:
    """
    Analyze pie chart using contour detection approach.
    
    This method uses color contour detection to find pie segments instead of
    the traditional crop-detect-measure approach. It's more robust for charts
    where traditional segmentation struggles.
    
    Args:
        img: Input image as numpy array
        segment_colors: Array of BGR colors for each segment (from SEGMENT_COLORS or extracted)
        image_id: Unique identifier for the image
        image_name: Name of the image
        sensitivity: Color matching sensitivity (default: 15)
        debug: Enable debug mode for verbose output
        
    Returns:
        Dictionary with analysis results including:
        - success: Boolean indicating if analysis succeeded
        - id: Image identifier
        - name: Image name
        - date: Timestamp of analysis
        - segment_1 through segment_9: Percentage values for each segment (if successful)
    """
    if debug:
        print(f"[PieChart-Contours] Analyzing image: {image_name}")            

    color_schemes = [load_color_scheme(1),
                     load_color_scheme(2),
                     load_color_scheme(3)]
    sensitivities = [20, 20, 15]

    default_result = {
        "success": False,
        "id": image_id,
        "name": image_name,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Try each color scheme until one succeeds
    for scheme_idx, segment_colors in enumerate(color_schemes):
        if segment_colors is None:
            if debug:
                print(f"[PieChart-Contours] Color scheme {scheme_idx + 1} not available, skipping")
            continue
        
        # Convert RGB to BGR if needed (OpenCV uses BGR format)
        #segment_colors = np.array([color[::-1] for color in segment_colors])
        sensitivity = sensitivities[scheme_idx]
        
        if debug:
            print(f"[PieChart-Contours] Trying color scheme {scheme_idx + 1} with sensitivity={sensitivity}")
        
        # Step 1: Detect ALL candidate contours for each color
        color_candidates = []  # List of lists: [[candidates for color1], [candidates for color2], ...]
        
        for i, color in enumerate(segment_colors):
            if debug:
                print(f"[PieChart-Contours]   Detecting segment {i+1} with color BGR={list(color)}")
            
            cnts = detect_color_contours(
                img=img,
                target_bgr=color,
                sensitivity=sensitivity,
                debug_mode=debug
            )
            
            # Filter for valid pie segments
            valid_segments = [c for c in cnts if is_pie_segment(c, 100)] if cnts else []
            color_candidates.append(valid_segments)
            
            if debug:
                print(f"[PieChart-Contours]     Found {len(valid_segments)} valid segment candidates")    
                if debug and valid_segments:
                    debug_img = img.copy()
                    cv2.drawContours(debug_img, valid_segments, -1, (0, 255, 0), 2)
                    cv2.imshow(f"Valid Segments for Color {i+1}", debug_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        
        # Step 2: Try different starting points and find the combination with tightest clustering
        best_set = None
        best_indices = None
        best_score = float('inf')
        
        for i, candidates in enumerate(color_candidates):
            if not candidates:
                continue
            
            # Try the largest candidate from this color as starting point
            largest = max(candidates, key=cv2.contourArea)
            segment_set, indices = build_segment_set(i, color_candidates, largest)
            
            if len(segment_set) + 2 >= len(segment_colors):
                score = calculate_clustering_score(segment_set)
                
                if debug:
                    print(f"[PieChart-Contours]   Try starting with color {i+1}: found {len(segment_set)} segments, score={score:.1f}")
                
                if score < best_score:
                    best_score = score
                    best_set = segment_set
                    best_indices = indices
        
        if best_set is None:
            if debug:
                print(f"[PieChart-Contours] Color scheme {scheme_idx + 1} failed - could not find complete segment set")
            continue  # Try next color scheme
        
        selected_contours = best_set
        selected_indices = best_indices
        
        if debug:
            print(f"[PieChart-Contours] Selected best combination with clustering score={best_score:.1f}")
        
        # Step 3: Build results with proper ordering
        all_contours = [None] * len(segment_colors)
        segment_areas = [0] * len(segment_colors)
        segment_bounds = [None] * len(segment_colors)
        
        for idx, contour in zip(selected_indices, selected_contours):
            all_contours[idx] = contour
            segment_areas[idx] = cv2.contourArea(contour)
            segment_bounds[idx] = cv2.boundingRect(contour)
        
        # Step 4: Check if we found enough segments
        detected_segments = sum(1 for area in segment_areas if area > 0)
        
        if detected_segments < len(segment_colors) - 2:
            if debug:
                print(f"[PieChart-Contours] Color scheme {scheme_idx + 1} failed - missing segments")
            continue  # Try next color scheme

        #if detected_segments < len(segment_colors):
        #    if debug:
        #        print(f"[PieChart-Contours] Color scheme {scheme_idx + 1} failed - only detected {detected_segments}/{len(segment_colors)} segments")
        #    continue  # Try next color scheme
        
        # Step 5: Draw detected segments for visualization
        if debug:
            debug_img = img.copy()
            for i, contour in enumerate(all_contours):
                if contour is not None:
                    color = tuple(map(int, segment_colors[i]))
                    cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)
                    # Draw segment number
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(debug_img, str(i+1), (cx, cy), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow(f"Detected Segments - Scheme {scheme_idx + 1} - {image_name}", debug_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Find pie chart center
        pie_center = None
        if detected_segments >= 7:
            if debug:
                print(f"[PieChart-Contours] Finding pie chart center from {detected_segments} segments")
            
            # Collect all line parameters from segment radial edges
            lines_params = []
            all_triangles = []  # Store all triangles to find inner apex later
            
            for i, contour in enumerate(all_contours):
                if contour is None:
                    continue
                # Approximate contour to exactly 3 vertices (triangle)
                # Use minEnclosingTriangle which gives the best fitting triangle
                _, triangle = cv2.minEnclosingTriangle(contour)
                
                if triangle is not None and len(triangle) == 3:
                    # Extract the 3 vertices and convert to integer tuples
                    p1 = (int(triangle[0][0][0]), int(triangle[0][0][1]))
                    p2 = (int(triangle[1][0][0]), int(triangle[1][0][1]))
                    p3 = (int(triangle[2][0][0]), int(triangle[2][0][1]))
                    all_triangles.append(((p1, p2, p3), i))
            
            # Now identify the inner apex for each triangle by finding the point
            # with smallest sum of distances to all other triangles' points
            for triangle, seg_idx in all_triangles:
                p1, p2, p3 = triangle
                
                # Collect all other triangles' points
                other_points = []
                for other_tri, other_idx in all_triangles:
                    if other_idx != seg_idx:
                        other_points.extend(other_tri)
                
                # Calculate sum of distances from each vertex to all other points
                if len(other_points) > 0:
                    sum_dist_p1 = sum(np.sqrt((p1[0] - op[0])**2 + (p1[1] - op[1])**2) for op in other_points)
                    sum_dist_p2 = sum(np.sqrt((p2[0] - op[0])**2 + (p2[1] - op[1])**2) for op in other_points)
                    sum_dist_p3 = sum(np.sqrt((p3[0] - op[0])**2 + (p3[1] - op[1])**2) for op in other_points)
                    
                    # The point with smallest sum is closest to all other segments (inner apex)
                    distances = [(sum_dist_p1, p1), (sum_dist_p2, p2), (sum_dist_p3, p3)]
                    distances.sort(key=lambda x: x[0])
                    
                    inner_apex = distances[0][1]
                    corner1 = distances[1][1]
                    corner2 = distances[2][1]
                else:
                    # Fallback: use smallest angle method
                    # Calculate angles at each vertex
                    def angle_at_vertex(va, vb, vc):
                        # Angle at vb between va and vc
                        v1 = (va[0] - vb[0], va[1] - vb[1])
                        v2 = (vc[0] - vb[0], vc[1] - vb[1])
                        dot = v1[0]*v2[0] + v1[1]*v2[1]
                        det = v1[0]*v2[1] - v1[1]*v2[0]
                        angle = np.arctan2(det, dot)
                        return abs(angle)
                    
                    angle_p1 = angle_at_vertex(p2, p1, p3)
                    angle_p2 = angle_at_vertex(p1, p2, p3)
                    angle_p3 = angle_at_vertex(p1, p3, p2)
                    
                    # Inner apex has the smallest angle
                    angles = [(angle_p1, p1), (angle_p2, p2), (angle_p3, p3)]
                    angles.sort(key=lambda x: x[0])
                    
                    inner_apex = angles[0][1]
                    corner1 = angles[1][1]
                    corner2 = angles[2][1]
                
                # Create radial lines from inner apex through each corner
                for corner in [corner1, corner2]:
                    dx = corner[0] - inner_apex[0]
                    dy = corner[1] - inner_apex[1]
                    
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        dx, dy = dx / length, dy / length
                        
                        # Line equation: dy*x - dx*y + (dx*inner_y - dy*inner_x) = 0
                        a = dy
                        b = -dx
                        c = dx * inner_apex[1] - dy * inner_apex[0]
                        
                        # Normalize
                        norm = np.sqrt(a**2 + b**2)
                        if norm > 0:
                            a, b, c = a/norm, b/norm, c/norm
                            lines_params.append((a, b, c, inner_apex, corner, seg_idx))
            
            if debug:
                print(f"[PieChart-Contours]   Extracted {len(lines_params)} radial lines from segment edges")                        
                        
            # Calculate center from inner apex points
            if len(lines_params) >= 6:  # At least 3 segments with 2 lines each
                # Extract unique inner apex points (each segment contributes 2 lines with same apex)
                inner_points = []
                seen_segments = set()
                
                for a, b, c, inner_pt, corner_pt, seg_idx in lines_params:
                    if seg_idx not in seen_segments:
                        inner_points.append(inner_pt)
                        seen_segments.add(seg_idx)
                
                # Ensure all inner apexes are clustered together (no outliers)
                if len(inner_points) >= 3:
                    inner_points_array = np.array(inner_points)
                    center_x = int(np.mean(inner_points_array[:, 0]))
                    center_y = int(np.mean(inner_points_array[:, 1]))
                    pie_center = (center_x, center_y)

                    # Calculate distances from center to each inner apex
                    distances = np.sqrt((inner_points_array[:, 0] - center_x) ** 2 + (inner_points_array[:, 1] - center_y) ** 2)
                    avg_distance = np.mean(distances)
                    max_distance = np.max(distances)
                    min_distance = np.min(distances)

                    # If any inner apex is much farther than the average (e.g., >2x avg), reject as not a pie chart
                    if max_distance > 3.0 * avg_distance:
                        if debug:
                            print(f"[PieChart-Contours]   ✗ Inner apexes are not clustered (max_distance={max_distance:.2f}, avg_distance={avg_distance:.2f})")
                        continue  # Try next color scheme
                    
                if len(inner_points) >= 3:
                    # Calculate center as centroid (mean) of inner apex points
                    inner_points_array = np.array(inner_points)
                    center_x = int(np.mean(inner_points_array[:, 0]))
                    center_y = int(np.mean(inner_points_array[:, 1]))
                    pie_center = (center_x, center_y)
                    
                    if debug:
                        # Calculate average distance from center to each inner point
                        distances = [np.sqrt((pt[0] - center_x)**2 + (pt[1] - center_y)**2) 
                                   for pt in inner_points]
                        avg_distance = np.mean(distances)
                        max_distance = np.max(distances)                                           
                            
                        print(f"[PieChart-Contours]   ✓ Pie chart center found at: {pie_center}")
                        print(f"[PieChart-Contours]   From {len(inner_points)} inner apex points")
                        print(f"[PieChart-Contours]   Average distance: {avg_distance:.2f} pixels")
                        print(f"[PieChart-Contours]   Max distance: {max_distance:.2f} pixels")
                    
                    # Calculate bisectors for ALL 9 segments
                    # Each segment is 360/9 = 40 degrees
                    # We need to find the starting angle first by looking at detected segments
                    segment_corners = {}
                    for a, b, c, inner_pt, corner_pt, seg_idx in lines_params:
                        if seg_idx not in segment_corners:
                            segment_corners[seg_idx] = []
                        segment_corners[seg_idx].append(corner_pt)
                    
                    # Calculate bisector angles for detected segments to find rotation
                    detected_bisector_angles = {}
                    for seg_idx, corners in segment_corners.items():
                        if len(corners) == 2:
                            angle1 = np.arctan2(corners[0][1] - pie_center[1], corners[0][0] - pie_center[0])
                            angle2 = np.arctan2(corners[1][1] - pie_center[1], corners[1][0] - pie_center[0])
                            
                            # Calculate bisector angle (handle wrap-around)
                            angle_diff = angle2 - angle1
                            if angle_diff > np.pi:
                                angle_diff -= 2 * np.pi
                            elif angle_diff < -np.pi:
                                angle_diff += 2 * np.pi
                            
                            bisector_angle = angle1 + angle_diff / 2
                            detected_bisector_angles[seg_idx] = bisector_angle
                    
                    # Find the starting angle (rotation offset) from detected segments
                    # Use the first detected segment's bisector as reference
                    if len(detected_bisector_angles) > 0:
                        first_seg_idx = min(detected_bisector_angles.keys())
                        first_bisector = detected_bisector_angles[first_seg_idx]
                        
                        # Expected angle for this segment index (assuming 40° per segment)
                        expected_angle = first_seg_idx * (2 * np.pi / 9)  # Convert to radians
                        
                        # Calculate rotation offset
                        rotation_offset = first_bisector - expected_angle
                        
                        if debug:
                            print(f"[PieChart-Contours]   Rotation offset: {np.degrees(rotation_offset):.2f}°")
                    else:
                        rotation_offset = 0
                    
                    # Now create bisectors for ALL 9 segments
                    segment_bisectors = []
                    max_length = max(img.shape[0], img.shape[1])
                    
                    for seg_idx in range(9):
                        # Calculate bisector angle for this segment
                        bisector_angle = rotation_offset + seg_idx * (2 * np.pi / 9)
                        
                        # Calculate bisector endpoint
                        bisector_end = (
                            int(pie_center[0] + max_length * np.cos(bisector_angle)),
                            int(pie_center[1] + max_length * np.sin(bisector_angle))
                        )
                        
                        segment_bisectors.append((seg_idx, pie_center, bisector_end, bisector_angle))
                    
                    if debug:
                        print(f"[PieChart-Contours]   Calculated {len(segment_bisectors)} segment bisectors")
                    
                    # Draw the center point and lines on debug image
                    if debug:
                        # Create a fresh image without contours
                        debug_center_img = img.copy()
                        
                        cv2.circle(debug_center_img, pie_center, 3, (0, 0, 255), -1)  # Red filled circle
                        
                        # Draw the bisectors
                        for seg_idx, start, end, angle in segment_bisectors:
                            cv2.line(debug_center_img, start, end, (0, 255, 0), 2)  # Green bisector lines
                            # Draw segment number along the bisector
                            text_pos = (
                                int(pie_center[0] + 50 * np.cos(angle)),
                                int(pie_center[1] + 50 * np.sin(angle))
                            )
                            cv2.putText(debug_center_img, str(seg_idx + 1), text_pos,
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Draw the radial lines from center through the corners
                        for a, b, c, inner_pt, corner_pt, seg_idx in lines_params:
                            #cv2.line(debug_center_img, pie_center, tuple(corner_pt), (255, 0, 255), 1)  # Magenta edge lines
                            
                            # Draw the inner apex points
                            cv2.circle(debug_center_img, tuple(inner_pt), 5, (0, 255, 255), -1)  # Yellow dots
                            
                            # Draw the corner points
                            cv2.circle(debug_center_img, tuple(corner_pt), 5, (255, 255, 0), -1)  # Cyan dots
                        
                        cv2.imshow(f"Center Detection - Scheme {scheme_idx + 1} - {image_name}", debug_center_img)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                else:
                    if debug:
                        print(f"[PieChart-Contours]   ✗ Not enough inner points extracted ({len(inner_points)} < 3)")
            else:
                if debug:
                    print(f"[PieChart-Contours]   ✗ Not enough lines extracted ({len(lines_params)} < 6)")
        

        # Step 6: Calculate percentages using bisector-based pixel analysis (for exploded pie charts)
        # For each bisector, walk from the center outward and detect where colored pixels end for each segment

        segment_lengths = [0] * 9
        max_length = max(img.shape[0], img.shape[1])
        img_bgr = img.copy()

        # First, estimate the smallest possible segment length (for offset)
        min_radius = min(pie_center[0], pie_center[1], img_bgr.shape[1] - pie_center[0], img_bgr.shape[0] - pie_center[1])
        #offset = int(0.5 * min_radius)
        offset = int(segment_bounds[0][2] / 4) if segment_bounds[0] is not None else int(0.1 * min_radius)

        for seg_idx, center, bisector_end, angle in segment_bisectors:
            found = False
            colors_along_bisector = []
            prev_bgr = None
            for r in range(offset, max_length):
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                if x < 0 or x >= img_bgr.shape[1] or y < 0 or y >= img_bgr.shape[0]:
                    segment_lengths[seg_idx] = r
                    found = True
                    break
                bgr = img_bgr[y, x].tolist()
                is_very_bright = all(v >= 240 for v in bgr)
                is_very_dark = all(v <= 40 for v in bgr)
                colors_along_bisector.append(bgr)
                if prev_bgr is not None:
                    color_jump = np.linalg.norm(np.array(bgr) - np.array(prev_bgr))
                    if is_very_bright or is_very_dark:
                        segment_lengths[seg_idx] = r
                        found = True                        
                        if debug:
                            print(f"Bisector {seg_idx+1}: Rapid change at r={r} (Δ={color_jump:.1f}, bright={is_very_bright}, dark={is_very_dark})")                            
                        break
                prev_bgr = bgr
            if not found:
                segment_lengths[seg_idx] = max_length

            if debug:
                # Print the colors found along this bisector
                print(f"Bisector {seg_idx+1}: {len(colors_along_bisector)} colors sampled.")
                for idx, color in enumerate(colors_along_bisector):
                    print(f"  r={offset+idx}: BGR={color}")

        # Calculate area-like value for each segment (proportional to length)
        # For a pie segment, area ~ (r2^2 - r1^2) * angle, but for exploded, we use length as proxy
        segment_areas_bisector = [l for l in segment_lengths]
        total_length = sum(segment_areas_bisector)
        total_area = sum((40/360*np.pi*length*length) for length in segment_areas_bisector)
        
        if total_length == 0:
            if debug:
                print(f"[PieChart-Contours] Bisector analysis failed - total length is zero")
            continue

        result = {
            "success": True,
            "id": image_id,
            "name": image_name,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "color_scheme_used": scheme_idx + 1,
        }

        for i, length in enumerate(segment_areas_bisector, start=1):
            #percentage = (length / total_length) * 100 if total_length > 0 else 0
            percentage = (40/360*np.pi*length*length) * 100 / total_area if total_area > 0 else 0
            result[f"segment_{i}"] = round(percentage, 2)
            if debug:
                print(f"[PieChart-Contours]   Segment {i}: {percentage:.2f}% (length={length})")

        if debug:
            print(f"[PieChart-Contours] ✓ Successfully analyzed image using bisector-based method, color scheme {scheme_idx + 1}: {image_name}")

        return result  # Success! Return immediately
    
    # If we get here, all color schemes failed
    if debug:
        print(f"[PieChart-Contours] ✗ All color schemes failed for image: {image_name}")
    
    return default_result
