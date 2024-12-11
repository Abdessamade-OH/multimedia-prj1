const express = require('express');
const multer = require('multer');
const path = require('path');
const Image = require('../models/Image');
const fs = require('fs');

const router = express.Router();

// Valid categories
const validCategories = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'];

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, './uploaded_images'); // Save files to uploaded_images folder
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`); // Unique file names
  }
});
const upload = multer({ storage });

router.post('/upload', upload.array('images', 400), async (req, res) => {
  try {
    const files = req.files;
    const { category } = req.body;

    // Check if category is provided and if it's valid
    if (!category) {
      return res.status(400).json({ error: 'Category is required.' });
    }

    // Check if category is one of the valid ones
    if (!validCategories.includes(category)) {
      return res.status(400).json({
        error: `Invalid category. Please choose from: ${validCategories.join(', ')}`,
      });
    }

    // Prepare image documents with category
    const imageDocs = files.map(file => ({
      name: file.originalname,
      category: category,  // Category is now mandatory and validated
      path: `/uploaded_images/${file.filename}`,
    }));

    // Insert all images into the database
    await Image.insertMany(imageDocs, { ordered: false });

    // Send success response
    res.status(201).json({ message: 'Images uploaded successfully!', images: imageDocs });
  } catch (err) {
    // Handle errors
    console.error(err); // Log error for debugging
    if (err.code === 11000) {
      res.status(400).json({ error: 'Duplicate image name detected. Please ensure all image names are unique.' });
    } else {
      res.status(500).json({ error: 'Error uploading images' });
    }
  }
});
  
// Endpoint to fetch all images
router.get('/', async (req, res) => {
  try {
    const images = await Image.find();
    res.status(200).json(images);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error fetching images' });
  }
});

// Endpoint to delete an image by its original name
router.delete('/delete/:name', async (req, res) => {
  try {
    const { name } = req.params;
    console.log(`Attempting to delete image with name: ${name}`);
    
    // Find the image in the database by its original name
    const image = await Image.findOne({ name });
    if (!image) {
      console.log(`Image not found in database with name: ${name}`);
      return res.status(404).json({ error: 'Image not found' });
    }

    // Get the full file path (unique file name in the filesystem)
    const filePath = path.join(__dirname, '../', image.path);
    console.log(`File path to delete: ${filePath}`);

    // Check if the file exists before trying to delete it
    if (!fs.existsSync(filePath)) {
      console.log(`File does not exist at path: ${filePath}`);
      return res.status(404).json({ error: 'File not found on server' });
    }

    // Delete the image file from the file system
    fs.unlinkSync(filePath);  // Delete the image file from the server
    console.log(`File deleted successfully: ${filePath}`);

    // Remove the image document from MongoDB
    await Image.findOneAndDelete({ name });
    console.log(`Image document deleted from database: ${name}`);

    res.status(200).json({ message: 'Image deleted successfully!' });
  } catch (err) {
    console.error('Error during image deletion:', err);
    res.status(500).json({ error: 'Error deleting image' });
  }
});

router.get('/all', async (req, res) => {
  try {
    const images = await Image.find();
    res.status(200).json(images);
  } catch (err) {
    res.status(500).json({ error: 'Error fetching images' });
  }
});

module.exports = router;
