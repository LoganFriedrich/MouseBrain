// RGB Direct Extraction Line Generator
// Extracts RGB component values directly from pixels without channel splitting

macro "RGB Direct Line Generator" {
    
    // Check if images are open
    if (nImages == 0) {
        showMessage("Error", "Please open at least one image first.");
        exit();
    }
    
    // Get list of all open images
    image_list = getList("image.titles");
    num_images = image_list.length;
    
    print("=== RGB DIRECT EXTRACTION LINE GENERATOR ===");
    print("Found " + num_images + " open images to process");
    
    // Test first image to see if it's RGB
    first_image = image_list[0];
    selectWindow(first_image);
    
    is_rgb = false;
    if (bitDepth() == 24) {
        is_rgb = true;
        print("Detected RGB color images (24-bit)");
    } else {
        print("Detected grayscale images (" + bitDepth() + "-bit)");
    }
    
    // Create dialog for RGB channel selection
    Dialog.create("RGB Direct Extraction");
    Dialog.addMessage("Select which RGB channels to extract as intensity lines:");
    
    if (is_rgb) {
        Dialog.addCheckbox("Red Component Intensities", true);
        Dialog.addCheckbox("Green Component Intensities", false);
        Dialog.addCheckbox("Blue Component Intensities", false);
        Dialog.addMessage("Note: Extracts RGB values directly from original image");
        Dialog.addCheckbox("Combine selected channels side-by-side", true);
    } else {
        Dialog.addCheckbox("Process as Grayscale", true);
        Dialog.addChoice("Color scheme:", newArray("Grayscale", "Red Scale", "Green Scale", "Blue Scale"));
    }
    
    Dialog.addMessage(" ");
    Dialog.addNumber("Line width (pixels):", 20);
    Dialog.addNumber("Width/Height ratio (e.g., 0.01 = 1:100):", 0.01);
    Dialog.addCheckbox("Show detailed statistics", true);
    Dialog.show();
    
    // Get settings
    process_red = false;
    process_green = false;
    process_blue = false;
    process_gray = false;
    gray_color = "";
    combine_channels = false;
    show_stats = false;
    
    if (is_rgb) {
        process_red = Dialog.getCheckbox();
        process_green = Dialog.getCheckbox();
        process_blue = Dialog.getCheckbox();
        combine_channels = Dialog.getCheckbox();
    } else {
        process_gray = Dialog.getCheckbox();
        gray_color = Dialog.getChoice();
    }
    
    line_width = Dialog.getNumber();
    width_ratio = Dialog.getNumber();
    show_stats = Dialog.getCheckbox();
    
    // Check if any channels selected
    channels_selected = 0;
    if (process_red) channels_selected++;
    if (process_green) channels_selected++;
    if (process_blue) channels_selected++;
    if (process_gray) channels_selected++;
    
    if (channels_selected == 0) {
        showMessage("Error", "Please select at least one channel to process.");
        exit();
    }
    
    print("Settings:");
    if (process_red) print("- Red component → Red Scale");
    if (process_green) print("- Green component → Green Scale");
    if (process_blue) print("- Blue component → Blue Scale");
    if (process_gray) print("- Grayscale → " + gray_color);
    print("- Line width: " + line_width);
    print("- Width/Height ratio: " + width_ratio);
    
    // Process each image
    for (img_idx = 0; img_idx < num_images; img_idx++) {
        
        current_image_name = image_list[img_idx];
        
        // Check if image is still open
        if (!isOpen(current_image_name)) {
            print("Skipping " + current_image_name + " - image was closed");
            continue;
        }
        
        print("\\n==================================================");
        print("Processing image " + (img_idx + 1) + " of " + num_images + ": " + current_image_name);
        print("==================================================");
        
        // Select the current image
        selectWindow(current_image_name);
        original_image = getTitle();
        
        // Get image info
        getDimensions(img_width, img_height, channels, slices, frames);
        print("Image dimensions: " + img_width + "x" + img_height);
        print("Bit depth: " + bitDepth());
        
        // Get image directory for saving
        image_dir = getInfo("image.directory");
        if (image_dir == "") image_dir = getInfo("user.dir") + File.separator;
        
        // Instructions for this specific image
        showMessage("Image " + (img_idx + 1) + " of " + num_images, 
            "Processing: " + original_image + "\\n\\n" +
            "Select a region with visible intensity variation across rows\\n\\n" +
            "1. Click CENTER TOP of your region\\n" +
            "2. Click CENTER BOTTOM of your region\\n\\n" +
            "Click OK to start point selection");
        
        // Get first point (center top)
        setTool("point");
        waitForUser("Click Center Top", "Image: " + original_image + "\\n\\nClick CENTER TOP of your analysis region\\nThen click OK");
        
        if (selectionType() != 10) {
            print("No point selected for " + original_image + ", skipping");
            continue;
        }
        
        getSelectionBounds(x1, y1, w1, h1);
        center_top_x = x1;
        center_top_y = y1;
        
        // Get second point (center bottom)
        run("Select None");
        waitForUser("Click Center Bottom", "Image: " + original_image + "\\n\\nClick CENTER BOTTOM of the same region\\nThen click OK");
        
        if (selectionType() != 10) {
            print("No bottom point selected for " + original_image + ", skipping");
            continue;
        }
        
        getSelectionBounds(x2, y2, w2, h2);
        center_bottom_x = x2;
        center_bottom_y = y2;
        
        // Calculate rectangle dimensions
        height = center_bottom_y - center_top_y;
        if (height <= 0) {
            print("Invalid points for " + original_image + " (bottom not below top), skipping");
            continue;
        }
        
        width = Math.round(height * width_ratio);
        center_x = (center_top_x + center_bottom_x) / 2;
        x = Math.round(center_x - (width / 2));
        y = center_top_y;
        
        print("Analysis rectangle:");
        print("  Position: x=" + x + ", y=" + y);
        print("  Size: " + width + "x" + height);
        print("  Center line: x=" + center_x);
        
        // Validate rectangle bounds
        if (x < 0 || y < 0 || x + width > img_width || y + height > img_height) {
            print("ERROR: Rectangle extends outside image bounds!");
            print("  Image: " + img_width + "x" + img_height);
            print("  Rectangle: " + x + "," + y + " to " + (x + width) + "," + (y + height));
            continue;
        }
        
        // Show rectangle for confirmation
        selectWindow(original_image);
        makeRectangle(x, y, width, height);
        waitForUser("Confirm Rectangle", 
            "Rectangle: " + width + "x" + height + " at (" + x + "," + y + ")\\n\\n" +
            "This will analyze " + height + " rows of " + width + " pixels each\\n\\n" +
            "OK to process, Cancel to skip");
        
        if (selectionType() != 0) {
            print("Rectangle selection lost, skipping");
            continue;
        }
        
        // Extract RGB component data directly from original image
        red_intensities = newArray();
        green_intensities = newArray();
        blue_intensities = newArray();
        gray_intensities = newArray();
        
        if (is_rgb) {
            // Extract RGB components directly
            if (process_red) {
                red_intensities = extractRGBComponent(original_image, "red", x, y, width, height, show_stats);
            }
            if (process_green) {
                green_intensities = extractRGBComponent(original_image, "green", x, y, width, height, show_stats);
            }
            if (process_blue) {
                blue_intensities = extractRGBComponent(original_image, "blue", x, y, width, height, show_stats);
            }
        } else {
            // Process grayscale
            if (process_gray) {
                gray_intensities = extractGrayscaleIntensities(original_image, x, y, width, height, show_stats);
            }
        }
        
        // Create line images
        base_name = replace(original_image, ".tif", "");
        base_name = replace(base_name, ".tiff", "");
        base_name = replace(base_name, ".png", "");
        base_name = replace(base_name, ".jpg", "");
        
        if (combine_channels && is_rgb && channels_selected > 1) {
            // Create combined side-by-side image
            createCombinedImage(red_intensities, green_intensities, blue_intensities, 
                              process_red, process_green, process_blue,
                              height, line_width, base_name, image_dir, show_stats);
        } else {
            // Create separate images
            if (process_red && red_intensities.length > 0) {
                createLineImage(red_intensities, "red", "Red Scale", height, line_width, base_name, image_dir, show_stats);
            }
            if (process_green && green_intensities.length > 0) {
                createLineImage(green_intensities, "green", "Green Scale", height, line_width, base_name, image_dir, show_stats);
            }
            if (process_blue && blue_intensities.length > 0) {
                createLineImage(blue_intensities, "blue", "Blue Scale", height, line_width, base_name, image_dir, show_stats);
            }
            if (process_gray && gray_intensities.length > 0) {
                createLineImage(gray_intensities, "gray", gray_color, height, line_width, base_name, image_dir, show_stats);
            }
        }
        
        print("Completed " + original_image + " - " + channels_selected + " channels processed");
    }
    
    print("\\n=== PROCESSING COMPLETE ===");
    showMessage("Complete!", "Processed " + num_images + " images with direct RGB extraction!");
}

// Function to extract RGB component values directly from original image
function extractRGBComponent(image_name, component, x, y, width, height, show_stats) {
    
    selectWindow(image_name);
    
    print("\\nExtracting " + component + " component directly from original image:");
    print("  Region: x=" + x + ", y=" + y + ", size=" + width + "x" + height);
    print("  Total pixels per row: " + width);
    print("  Total rows to process: " + height);
    
    // Array to store row averages
    row_intensities = newArray(height);
    min_intensity = 999999;
    max_intensity = 0;
    total_sum = 0;
    
    // Process each row
    for (row = 0; row < height; row++) {
        current_y = y + row;
        row_sum = 0;
        pixel_count = 0;
        
        // Process each pixel in this row
        for (col = 0; col < width; col++) {
            current_x = x + col;
            
            // Get RGB value at this pixel
            rgb_value = getPixel(current_x, current_y);
            
            // Extract the desired component
            component_value = 0;
            if (component == "red") {
                component_value = (rgb_value >> 16) & 0xff;
            } else if (component == "green") {
                component_value = (rgb_value >> 8) & 0xff;
            } else if (component == "blue") {
                component_value = rgb_value & 0xff;
            }
            
            row_sum += component_value;
            pixel_count++;
        }
        
        // Calculate average for this row
        row_average = row_sum / pixel_count;
        row_intensities[row] = row_average;
        total_sum += row_average;
        
        // Track min/max
        if (row_average < min_intensity) min_intensity = row_average;
        if (row_average > max_intensity) max_intensity = row_average;
        
        // Debug output for first few rows
        if (show_stats && row < 5) {
            print("    Row " + row + " (y=" + current_y + "): " + pixel_count + " pixels, average " + component + " = " + d2s(row_average, 2));
        }
        
        // Progress indicator for long extractions
        if (height > 100 && row % Math.floor(height / 10) == 0) {
            print("    Progress: " + Math.round((row / height) * 100) + "% (" + row + "/" + height + " rows)");
        }
    }
    
    overall_average = total_sum / height;
    intensity_range = max_intensity - min_intensity;
    
    print("  " + component + " component analysis complete:");
    print("    Min row average: " + d2s(min_intensity, 2));
    print("    Max row average: " + d2s(max_intensity, 2));
    print("    Range: " + d2s(intensity_range, 2));
    print("    Overall average: " + d2s(overall_average, 2));
    
    // Quality check
    if (intensity_range < 1) {
        print("    WARNING: Very low intensity range - line may appear uniform");
        print("    Consider selecting a region with more " + component + " variation");
    } else if (intensity_range > 20) {
        print("    GOOD: Intensity range should show clear variation");
    } else {
        print("    OK: Moderate intensity range should show some variation");
    }
    
    return row_intensities;
}

// Function to extract grayscale intensities directly
function extractGrayscaleIntensities(image_name, x, y, width, height, show_stats) {
    
    selectWindow(image_name);
    
    print("\\nExtracting grayscale intensities directly:");
    print("  Region: x=" + x + ", y=" + y + ", size=" + width + "x" + height);
    
    // Array to store row averages
    row_intensities = newArray(height);
    min_intensity = 999999;
    max_intensity = 0;
    total_sum = 0;
    
    // Process each row
    for (row = 0; row < height; row++) {
        current_y = y + row;
        row_sum = 0;
        
        // Process each pixel in this row
        for (col = 0; col < width; col++) {
            current_x = x + col;
            pixel_value = getPixel(current_x, current_y);
            row_sum += pixel_value;
        }
        
        // Calculate average for this row
        row_average = row_sum / width;
        row_intensities[row] = row_average;
        total_sum += row_average;
        
        // Track min/max
        if (row_average < min_intensity) min_intensity = row_average;
        if (row_average > max_intensity) max_intensity = row_average;
        
        if (show_stats && row < 5) {
            print("    Row " + row + ": average = " + d2s(row_average, 2));
        }
    }
    
    overall_average = total_sum / height;
    intensity_range = max_intensity - min_intensity;
    
    print("  Grayscale analysis complete:");
    print("    Min row average: " + d2s(min_intensity, 2));
    print("    Max row average: " + d2s(max_intensity, 2));
    print("    Range: " + d2s(intensity_range, 2));
    print("    Overall average: " + d2s(overall_average, 2));
    
    return row_intensities;
}

// Function to create a single line image from intensity values
function createLineImage(intensities, channel_name, color_scheme, height, line_width, base_name, image_dir, show_stats) {
    
    if (intensities.length != height) {
        print("ERROR: Intensity array length doesn't match height");
        return;
    }
    
    // Find min and max for normalization
    min_val = intensities[0];
    max_val = intensities[0];
    for (i = 0; i < intensities.length; i++) {
        if (intensities[i] < min_val) min_val = intensities[i];
        if (intensities[i] > max_val) max_val = intensities[i];
    }
    
    print("\\nCreating " + channel_name + " line image (" + line_width + "x" + height + ")");
    print("  Normalizing intensity range: " + d2s(min_val, 2) + " to " + d2s(max_val, 2));
    
    // Create new RGB image for the line
    newImage("Line_" + channel_name + "_" + base_name, "RGB", line_width, height, 1);
    
    // Fill each row with color based on its intensity
    for (row = 0; row < height; row++) {
        
        // Normalize intensity to 0-1 range
        if (max_val > min_val) {
            normalized = (intensities[row] - min_val) / (max_val - min_val);
        } else {
            normalized = 0.5; // All same value
        }
        
        // Convert to 0-255 color value
        color_value = Math.round(normalized * 255);
        
        // Set RGB values based on color scheme
        red = 0; green = 0; blue = 0;
        if (color_scheme == "Red Scale") {
            red = color_value;
        } else if (color_scheme == "Green Scale") {
            green = color_value;
        } else if (color_scheme == "Blue Scale") {
            blue = color_value;
        } else { // Grayscale
            red = green = blue = color_value;
        }
        
        // Fill this row
        setForegroundColor(red, green, blue);
        makeRectangle(0, row, line_width, 1);
        run("Fill");
        
        // Debug output for first few rows
        if (show_stats && row < 3) {
            print("    Row " + row + ": intensity=" + d2s(intensities[row], 2) + 
                  " → normalized=" + d2s(normalized, 3) + 
                  " → color=" + color_value + " → RGB(" + red + "," + green + "," + blue + ")");
        }
    }
    
    run("Select None");
    
    // Save the line image
    filename = image_dir + base_name + "_" + channel_name + "_line.png";
    saveAs("PNG", filename);
    print("  Saved: " + filename);
    
    // Keep image open for viewing
    rename(base_name + "_" + channel_name + "_line");
}

// Function to create combined side-by-side image
function createCombinedImage(red_intensities, green_intensities, blue_intensities, 
                           use_red, use_green, use_blue, height, line_width, base_name, image_dir, show_stats) {
    
    // Count channels to combine
    num_channels = 0;
    if (use_red && red_intensities.length > 0) num_channels++;
    if (use_green && green_intensities.length > 0) num_channels++;
    if (use_blue && blue_intensities.length > 0) num_channels++;
    
    if (num_channels == 0) return;
    
    total_width = line_width * num_channels;
    print("\\nCreating combined image: " + total_width + "x" + height);
    
    newImage("Combined_Lines_" + base_name, "RGB", total_width, height, 1);
    
    current_x = 0;
    channel_names = "";
    
    // Add each channel column
    if (use_red && red_intensities.length > 0) {
        fillChannelColumn(red_intensities, current_x, line_width, height, "red");
        current_x += line_width;
        channel_names += "red";
    }
    
    if (use_green && green_intensities.length > 0) {
        fillChannelColumn(green_intensities, current_x, line_width, height, "green");
        current_x += line_width;
        if (channel_names != "") channel_names += "_";
        channel_names += "green";
    }
    
    if (use_blue && blue_intensities.length > 0) {
        fillChannelColumn(blue_intensities, current_x, line_width, height, "blue");
        current_x += line_width;
        if (channel_names != "") channel_names += "_";
        channel_names += "blue";
    }
    
    run("Select None");
    
    // Save combined image
    filename = image_dir + base_name + "_" + channel_names + "_combined.png";
    saveAs("PNG", filename);
    print("  Saved combined: " + filename);
    
    // Keep open for viewing
    rename(base_name + "_" + channel_names + "_combined");
}

// Helper function to fill one column of combined image
function fillChannelColumn(intensities, x_start, width, height, color_name) {
    
    // Normalize intensities for this column
    min_val = intensities[0];
    max_val = intensities[0];
    for (i = 0; i < intensities.length; i++) {
        if (intensities[i] < min_val) min_val = intensities[i];
        if (intensities[i] > max_val) max_val = intensities[i];
    }
    
    print("  Filling " + color_name + " column: range " + d2s(min_val, 2) + " to " + d2s(max_val, 2));
    
    for (row = 0; row < height; row++) {
        // Normalize
        if (max_val > min_val) {
            normalized = (intensities[row] - min_val) / (max_val - min_val);
        } else {
            normalized = 0.5;
        }
        
        // Convert to color
        color_value = Math.round(normalized * 255);
        
        red = 0; green = 0; blue = 0;
        if (color_name == "red") {
            red = color_value;
        } else if (color_name == "green") {
            green = color_value;
        } else if (color_name == "blue") {
            blue = color_value;
        }
        
        setForegroundColor(red, green, blue);
        makeRectangle(x_start, row, width, 1);
        run("Fill");
    }
}