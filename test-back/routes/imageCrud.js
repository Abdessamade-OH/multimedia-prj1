const express = require('express');
const multer = require('multer');
const path = require('path');
const Image = require('../models/Image'); // Assuming this is a Mongoose model
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
    const uniqueFilename = `${Date.now()}-${file.originalname}`;
    cb(null, uniqueFilename); // Ensures unique filenames in storage
  }
});
const upload = multer({ storage });

// POST: Upload images
router.post('/upload', upload.array('images', 400), async (req, res) => {
  try {
    const files = req.files;
    const { category } = req.body;

    // Validate category
    if (!category) {
      return res.status(400).json({ error: 'Category is required.' });
    }

    if (!validCategories.includes(category)) {
      return res.status(400).json({
        error: `Invalid category. Please choose from: ${validCategories.join(', ')}`,
      });
    }

    // Create image documents
    const imageDocs = files.map(file => ({
      name: file.originalname, // Original name for user reference
      category,
      path: `/uploaded_images/${file.filename}`, // File path
    }));

    // Insert documents into the database
    const savedImages = await Image.insertMany(imageDocs);

    res.status(201).json({ message: 'Images uploaded successfully!', images: savedImages });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error uploading images' });
  }
});

// GET: Fetch all images
router.get('/all', async (req, res) => {
  try {
    const images = await Image.find();
    res.status(200).json(images);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error fetching images' });
  }
});

// Endpoint to fetch images by name
router.get('/name/:name', async (req, res) => {
  try {
    const { name } = req.params;

    // Find all images with the specified name
    const images = await Image.find({ name });

    if (!images.length) {
      return res.status(404).json({ error: 'No images found with the specified name.' });
    }

    res.status(200).json(images);
  } catch (err) {
    console.error('Error fetching images by name:', err);
    res.status(500).json({ error: 'Error fetching images' });
  }
});

// DELETE: Delete an image by MongoDB ObjectId
router.delete('/delete/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Find the image by its MongoDB ObjectId
    const image = await Image.findById(id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    // Get the file path
    const filePath = path.join(__dirname, '../', image.path);

    // Check if the file exists
    if (fs.existsSync(filePath)) {
      // Delete the file from the file system
      fs.unlinkSync(filePath);
    }

    // Delete the image document from the database
    await Image.findByIdAndDelete(id);

    res.status(200).json({ message: 'Image deleted successfully!' });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error deleting image' });
  }
});

// GET: Fetch an image by MongoDB ObjectId
router.get('/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Find the image by its MongoDB ObjectId
    const image = await Image.findById(id);
    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    res.status(200).json(image);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: 'Error fetching image' });
  }
});

module.exports = router;
