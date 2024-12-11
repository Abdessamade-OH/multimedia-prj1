const mongoose = require('mongoose');

const validCategories = [
  'aGrass', 'bField', 'cIndustry', 'dRiverLake', 'eForest', 'fResident', 'gParking'
];

const imageSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
    unique: true,
  },
  category: {
    type: String,
    required: [true, 'Category is required'],
    enum: {
      values: validCategories,
      message: '{VALUE} is not a valid category. Choose one of the following: ' + validCategories.join(', ')
    },
  },
  path: {
    type: String,
    required: true,
  },
});

module.exports = mongoose.model('Image', imageSchema);
