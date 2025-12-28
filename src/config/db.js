const mongoose = require('mongoose');

async function connectDatabase(uri) {
  if (!uri) {
    throw new Error('MongoDB connection string is missing. Provide MONGODB_URI in your .env file.');
  }

  try {
    await mongoose.connect(uri, {
      maxPoolSize: 10,
    });
    console.log('✅ Connected to MongoDB Atlas');
  } catch (error) {
    console.error('❌ MongoDB connection failed:', error.message);
    throw error;
  }
}

module.exports = {
  connectDatabase,
};
