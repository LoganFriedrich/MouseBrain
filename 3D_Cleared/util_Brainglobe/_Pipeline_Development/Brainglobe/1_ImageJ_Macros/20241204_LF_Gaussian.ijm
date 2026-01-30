// Prompt user to select source and destination directories
dir1 = getDirectory("Choose Source Directory ");
dir2 = getDirectory("Choose Destination Directory ");

// Get list of files in source directory
list = getFileList(dir1);

// Enable batch mode for faster processing
setBatchMode(true);

// Process each file in the directory
for (i=0; i<list.length; i++) {
    // Show progress
    showProgress(i+1, list.length);
    
    // Open current image
    open(dir1+list[i]);
    
    // Apply Gaussian blur
    // You can adjust the sigma value to control blur strength
    run("Gaussian Blur...", "sigma=2");
    
    // Save processed image
    saveAs("tiff", dir2+list[i]);
    
    // Close the image to free up memory
    close();
}

// Show completion message
showMessage("Macro Complete", "Processed " + list.length + " images");