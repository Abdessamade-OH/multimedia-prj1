const express = require('express');
const multer = require('multer');
const path = require('path');
const Image = require('../models/Image'); // Assuming this is a Mongoose model
const fs = require('fs');

const router = express.Router();

// Valid categories
const validCategories = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'];

// Middleware to parse multipart form data
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      // Extract category from the request - this is the key change
      const { category } = req.body;

      // Validate category
      const validCategories = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'];
      
      if (!category || !validCategories.includes(category)) {
        return cb(new Error(`Invalid category: ${category}`));
      }

      // Create the category-specific subfolder if it doesn't exist
      const uploadDir = path.join(__dirname, '../src/upload_folder', category);
      
      // Create directory if it doesn't exist
      fs.mkdirSync(uploadDir, { recursive: true });

      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const uniqueFilename = `${Date.now()}-${file.originalname}`;
      cb(null, uniqueFilename);
    }
  }),
  fileFilter: (req, file, cb) => {
    const { category } = req.body;
    
    const validCategories = ['aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'];
    
    // Validate category
    if (!category || !validCategories.includes(category)) {
      const error = new Error(`Invalid or missing category. Valid categories are: ${validCategories.join(', ')}`);
      return cb(error, false);
    }
    
    cb(null, true);
  }
}).array('images', 400);

// POST: Upload images
router.post('/upload', (req, res) => {
  // Wrap multer in a promise to handle errors
  new Promise((resolve, reject) => {
    upload(req, res, (err) => {
      if (err) {
        console.error('Upload error:', err);
        return reject(err);
      }
      resolve();
    });
  })
  .then(async () => {
    const files = req.files;
    const { category } = req.body;

    // Create image documents
    const imageDocs = files.map(file => ({
      name: file.originalname,
      category,
      path: `/src/upload_folder/${category}/${file.filename}`,
    }));

    // Insert documents into the database
    const savedImages = await Image.insertMany(imageDocs);

    res.status(201).json({ 
      message: 'Images uploaded successfully!', 
      images: savedImages 
    });
  })
  .catch(err => {
    console.error('Upload process error:', err);
    res.status(400).json({ 
      error: err.message || 'Error uploading images' 
    });
  });
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

// GET: Fetch images by category
router.get('/category/:category', async (req, res) => {
  try {
    const { category } = req.params;

    // Validate category (optional, based on your requirements)
    if (!validCategories.includes(category)) {
      return res.status(400).json({
        error: `Invalid category. Please choose from: ${validCategories.join(', ')}`,
      });
    }

    // Find images that match the category
    const images = await Image.find({ category });

    if (!images.length) {
      return res.status(404).json({ error: 'No images found in this category.' });
    }

    res.status(200).json(images);
  } catch (err) {
    console.error('Error fetching images by category:', err);
    res.status(500).json({ error: 'Error fetching images' });
  }
});

module.exports = router;
