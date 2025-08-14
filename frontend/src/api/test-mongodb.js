// Simple API endpoint to test MongoDB connectivity
import { MongoClient } from 'mongodb';
import express from 'express';

const router = express.Router();

router.get('/test-mongodb', async (req, res) => {
  const mongoUrl = process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  console.log(`Testing MongoDB connection at: ${mongoUrl}`);
  
  try {
    const client = new MongoClient(mongoUrl, {
      connectTimeoutMS: 5000,
      socketTimeoutMS: 5000
    });
    
    await client.connect();
    console.log('Successfully connected to MongoDB');
    
    const admin = client.db().admin();
    const dbInfo = await admin.listDatabases();
    const databases = dbInfo.databases.map(db => db.name);
    
    // Check for StockAI database
    let stockAIExists = databases.includes('StockAI');
    let collectionsInfo = [];
    
    if (stockAIExists) {
      const db = client.db('StockAI');
      const collections = await db.listCollections().toArray();
      collectionsInfo = collections.map(c => c.name);
    }
    
    await client.close();
    
    res.json({
      success: true,
      message: 'Successfully connected to MongoDB',
      databases: databases,
      stockAIExists: stockAIExists,
      collections: collectionsInfo
    });
  } catch (error) {
    console.error('Error connecting to MongoDB:', error);
    res.status(500).json({
      success: false,
      message: `Failed to connect to MongoDB: ${error.message}`,
      error: error.stack
    });
  }
});

export default router; 