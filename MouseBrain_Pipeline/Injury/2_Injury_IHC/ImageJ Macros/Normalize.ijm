// ImageJ Macro: Normalize Channel Brightness Across Images
// Makes red equally bright in all images, green equally bright, blue equally bright

fileExtension = ".ome.tif";

macro "Normalize Channel Brightness" {
    
    inputDir = getDirectory("Choose folder with your MIP .ome.tif images:");
    outputDir = getDirectory("Choose output folder for normalized images:");
    
    // Get list of files
    fileList = getFileList(inputDir);
    imageFiles = newArray();
    count = 0;
    
    for (i = 0; i < fileList.length; i++) {
        if (endsWith(fileList[i], fileExtension)) {
            imageFiles[count] = fileList[i];
            count++;
        }
    }
    
    print("Found " + count + " images to normalize");
    
    // STEP 1: Calculate mean intensities for each channel across all images
    redMeans = newArray(count);
    greenMeans = newArray(count);
    blueMeans = newArray(count);
    
    print("Calculating mean intensities...");
    
    for (i = 0; i < count; i++) {
        open(inputDir + imageFiles[i]);
        imageTitle = getTitle();
        
        if (bitDepth() != 24) {
            run("RGB Color");
        }
        run("Split Channels");
        
        // Get mean for Red
        selectWindow(imageTitle + " (red)");
        getStatistics(area, mean, min, max, std);
        redMeans[i] = mean;
        close();
        
        // Get mean for Green
        selectWindow(imageTitle + " (green)");
        getStatistics(area, mean, min, max, std);
        greenMeans[i] = mean;
        close();
        
        // Get mean for Blue
        selectWindow(imageTitle + " (blue)");
        getStatistics(area, mean, min, max, std);
        blueMeans[i] = mean;
        close();
        
        print("Image " + (i+1) + ": R=" + redMeans[i] + ", G=" + greenMeans[i] + ", B=" + blueMeans[i]);
    }
    
    // STEP 2: Find target intensities (use the brightest image as target)
    redTarget = redMeans[0];
    greenTarget = greenMeans[0];
    blueTarget = blueMeans[0];
    
    for (i = 1; i < count; i++) {
        if (redMeans[i] > redTarget) redTarget = redMeans[i];
        if (greenMeans[i] > greenTarget) greenTarget = greenMeans[i];
        if (blueMeans[i] > blueTarget) blueTarget = blueMeans[i];
    }
    
    print("Target intensities - R:" + redTarget + ", G:" + greenTarget + ", B:" + blueTarget);
    
    // STEP 3: Normalize all images to match target intensities
    print("Normalizing images...");
    
    for (i = 0; i < count; i++) {
        open(inputDir + imageFiles[i]);
        imageTitle = getTitle();
        
        if (bitDepth() != 24) {
            run("RGB Color");
        }
        run("Split Channels");
        
        // Calculate multiplication factors
        redFactor = redTarget / redMeans[i];
        greenFactor = greenTarget / greenMeans[i];
        blueFactor = blueTarget / blueMeans[i];
        
        print("Image " + (i+1) + " factors - R:" + redFactor + ", G:" + greenFactor + ", B:" + blueFactor);
        
        // Normalize Red channel
        selectWindow(imageTitle + " (red)");
        run("Multiply...", "value=" + redFactor);
        
        // Normalize Green channel
        selectWindow(imageTitle + " (green)");
        run("Multiply...", "value=" + greenFactor);
        
        // Normalize Blue channel
        selectWindow(imageTitle + " (blue)");
        run("Multiply...", "value=" + blueFactor);
        
        // Merge back to RGB
        run("Merge Channels...", "c1=[" + imageTitle + " (red)] c2=[" + imageTitle + " (green)] c3=[" + imageTitle + " (blue)]");
        
        // Save normalized image
        baseName = replace(imageFiles[i], fileExtension, "");
        saveAs("Tiff", outputDir + baseName + "_normalized.tif");
        close();
        
        print("Normalized: " + imageFiles[i]);
    }
    
    print("Brightness normalization complete!");
    print("All channels now have equal brightness across images");
}

// Alternative: Normalize to specific target values
macro "Normalize to Target Values" {
    
    inputDir = getDirectory("Choose folder:");
    outputDir = getDirectory("Choose output folder:");
    
    // SET YOUR TARGET BRIGHTNESS VALUES HERE
    targetRed = 2000;    // Desired red channel brightness
    targetGreen = 1800;  // Desired green channel brightness  
    targetBlue = 1500;   // Desired blue channel brightness
    
    fileList = getFileList(inputDir);
    
    for (i = 0; i < fileList.length; i++) {
        if (endsWith(fileList[i], ".ome.tif")) {
            open(inputDir + fileList[i]);
            imageTitle = getTitle();
            
            if (bitDepth() != 24) {
                run("RGB Color");
            }
            run("Split Channels");
            
            // Red channel
            selectWindow(imageTitle + " (red)");
            getStatistics(area, mean, min, max, std);
            redFactor = targetRed / mean;
            run("Multiply...", "value=" + redFactor);
            
            // Green channel
            selectWindow(imageTitle + " (green)");
            getStatistics(area, mean, min, max, std);
            greenFactor = targetGreen / mean;
            run("Multiply...", "value=" + greenFactor);
            
            // Blue channel
            selectWindow(imageTitle + " (blue)");
            getStatistics(area, mean, min, max, std);
            blueFactor = targetBlue / mean;
            run("Multiply...", "value=" + blueFactor);
            
            // Merge and save
            run("Merge Channels...", "c1=[" + imageTitle + " (red)] c2=[" + imageTitle + " (green)] c3=[" + imageTitle + " (blue)]");
            
            baseName = replace(fileList[i], ".ome.tif", "");
            saveAs("Tiff", outputDir + baseName + "_normalized.tif");
            close();
        }
    }
    
    print("Target normalization complete!");
}