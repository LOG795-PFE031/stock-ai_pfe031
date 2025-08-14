// Express server setup for API endpoints
import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import { MongoClient } from 'mongodb';
import dotenv from 'dotenv';

// Import route handlers
import testMongoDbRouter from './test-mongodb.js';
import conversationRouter from './ConversationApi.js';

// Load environment variables
dotenv.config();

// Create Express app
const app = express();
const PORT = process.env.PORT || 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));

// Logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', timestamp: new Date().toISOString() });
});

// API routes
app.use(testMongoDbRouter);
app.use(conversationRouter);

// Test MongoDB connection on startup
const testMongoDBConnection = async () => {
  const mongoUrl = process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  console.log(`Testing MongoDB connection at: ${mongoUrl}`);
  
  let client;
  try {
    client = new MongoClient(mongoUrl, {
      connectTimeoutMS: 5000,
      socketTimeoutMS: 5000
    });
    
    await client.connect();
    console.log('✅ Successfully connected to MongoDB!');
    
    // Get available databases
    const admin = client.db().admin();
    const dbInfo = await admin.listDatabases();
    console.log('Available databases:');
    dbInfo.databases.forEach(db => {
      console.log(`- ${db.name}`);
    });
    
    // Check for StockAI database
    const stockAIExists = dbInfo.databases.some(db => db.name === 'StockAI');
    console.log(`StockAI database exists: ${stockAIExists ? 'Yes' : 'No'}`);
    
    if (stockAIExists) {
      const db = client.db('StockAI');
      const collections = await db.listCollections().toArray();
      console.log('Collections in StockAI:');
      collections.forEach(c => {
        console.log(`- ${c.name}`);
      });
    }
    
  } catch (error) {
    console.error('❌ MongoDB connection error:', error);
  } finally {
    if (client) await client.close();
  }
};

// Start server
app.listen(PORT, async () => {
  console.log(`Server running on port ${PORT}`);
  await testMongoDBConnection();
});

export default app; 