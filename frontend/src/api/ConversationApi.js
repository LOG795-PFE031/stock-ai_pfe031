// API endpoints for conversation management with MongoDB
import { MongoClient } from 'mongodb';
import express from 'express';

const router = express.Router();

// Database and collection names
const DB_NAME = 'StockAI';
const CONVERSATIONS_COLLECTION = 'conversations';
const SUMMARIES_COLLECTION = 'summaries';

// Save a conversation message
router.post('/api/conversations', async (req, res) => {
  const { userId, message, sender, timestamp, mongoUrl } = req.body;
  
  if (!userId || !message || !sender) {
    return res.status(400).json({ error: 'Missing required fields' });
  }
  
  const dbUrl = mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  console.log(`Saving conversation for user ${userId} to MongoDB at ${dbUrl}`);
  
  let client;
  try {
    client = new MongoClient(dbUrl);
    await client.connect();
    
    const db = client.db(DB_NAME);
    const collection = db.collection(CONVERSATIONS_COLLECTION);
    
    const result = await collection.insertOne({
      userId,
      message,
      sender,
      timestamp: timestamp || new Date()
    });
    
    console.log(`Conversation saved with ID: ${result.insertedId}`);
    res.status(200).json({ success: true, messageId: result.insertedId });
  } catch (error) {
    console.error('Error saving conversation:', error);
    res.status(500).json({ error: `Failed to save conversation: ${error.message}` });
  } finally {
    if (client) await client.close();
  }
});

// Get conversation history for a user
router.get('/api/conversations/:userId', async (req, res) => {
  const { userId } = req.params;
  const mongoUrl = req.query.mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  
  console.log(`Fetching conversations for user ${userId} from MongoDB at ${mongoUrl}`);
  
  let client;
  try {
    client = new MongoClient(mongoUrl);
    await client.connect();
    
    const db = client.db(DB_NAME);
    const collection = db.collection(CONVERSATIONS_COLLECTION);
    
    const messages = await collection
      .find({ userId })
      .sort({ timestamp: 1 })
      .toArray();
    
    console.log(`Found ${messages.length} messages for user ${userId}`);
    res.status(200).json({ success: true, messages });
  } catch (error) {
    console.error('Error fetching conversations:', error);
    res.status(500).json({ error: `Failed to fetch conversations: ${error.message}` });
  } finally {
    if (client) await client.close();
  }
});

// Save a conversation summary
router.post('/api/conversations/:userId/summary', async (req, res) => {
  const { userId } = req.params;
  const { summary, timestamp, mongoUrl } = req.body;
  
  if (!summary) {
    return res.status(400).json({ error: 'Summary is required' });
  }
  
  const dbUrl = mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  console.log(`Saving summary for user ${userId} to MongoDB at ${dbUrl}`);
  
  let client;
  try {
    client = new MongoClient(dbUrl);
    await client.connect();
    
    const db = client.db(DB_NAME);
    const collection = db.collection(SUMMARIES_COLLECTION);
    
    // Save the new summary
    const result = await collection.insertOne({
      userId,
      summary,
      timestamp: timestamp || new Date()
    });
    
    console.log(`Summary saved with ID: ${result.insertedId}`);
    res.status(200).json({ success: true, summaryId: result.insertedId });
  } catch (error) {
    console.error('Error saving summary:', error);
    res.status(500).json({ error: `Failed to save summary: ${error.message}` });
  } finally {
    if (client) await client.close();
  }
});

// Get the latest conversation summary for a user
router.get('/api/conversations/:userId/summary', async (req, res) => {
  const { userId } = req.params;
  const mongoUrl = req.query.mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  
  console.log(`Fetching latest summary for user ${userId} from MongoDB at ${mongoUrl}`);
  
  let client;
  try {
    client = new MongoClient(mongoUrl);
    await client.connect();
    
    const db = client.db(DB_NAME);
    const collection = db.collection(SUMMARIES_COLLECTION);
    
    // Get the most recent summary
    const latestSummary = await collection
      .find({ userId })
      .sort({ timestamp: -1 })
      .limit(1)
      .toArray();
    
    if (latestSummary.length > 0) {
      console.log(`Found summary for user ${userId}`);
      res.status(200).json({ success: true, summary: latestSummary[0].summary });
    } else {
      console.log(`No summary found for user ${userId}`);
      res.status(404).json({ error: 'No summary found' });
    }
  } catch (error) {
    console.error('Error fetching summary:', error);
    res.status(500).json({ error: `Failed to fetch summary: ${error.message}` });
  } finally {
    if (client) await client.close();
  }
});

// Test MongoDB connection
router.post('/api/test-connection', async (req, res) => {
  const { userId, message, mongoUrl } = req.body;
  const dbUrl = mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  
  console.log(`Testing connection to MongoDB at ${dbUrl} with user ${userId}`);
  
  let client;
  try {
    client = new MongoClient(dbUrl);
    await client.connect();
    
    // Test write operation
    const db = client.db(DB_NAME);
    const collection = db.collection(CONVERSATIONS_COLLECTION);
    
    const result = await collection.insertOne({
      userId: userId || 'test-user',
      message: message || 'Connection test message',
      sender: 'system',
      timestamp: new Date()
    });
    
    console.log(`Test message saved with ID: ${result.insertedId}`);
    res.status(200).json({ 
      success: true, 
      message: 'MongoDB connection successful',
      messageId: result.insertedId 
    });
  } catch (error) {
    console.error('Error testing MongoDB connection:', error);
    res.status(500).json({ 
      success: false, 
      error: `MongoDB connection failed: ${error.message}` 
    });
  } finally {
    if (client) await client.close();
  }
});

// Test MongoDB connection and check database status
router.post('/api/test-mongodb', async (req, res) => {
  const { mongoUrl } = req.body;
  const dbUrl = mongoUrl || process.env.VITE_MONGODB_URL || 'mongodb://mongo:mongo@localhost:27017';
  
  console.log(`Testing MongoDB connection at: ${dbUrl}`);
  
  let client;
  try {
    client = new MongoClient(dbUrl, {
      connectTimeoutMS: 5000,
      socketTimeoutMS: 5000
    });
    
    await client.connect();
    console.log('Successfully connected to MongoDB');
    
    const admin = client.db().admin();
    const dbInfo = await admin.listDatabases();
    const databases = dbInfo.databases.map(db => db.name);
    
    // Check for StockAI database
    let stockAIExists = databases.includes(DB_NAME);
    let collectionsInfo = [];
    
    if (stockAIExists) {
      const db = client.db(DB_NAME);
      const collections = await db.listCollections().toArray();
      collectionsInfo = collections.map(c => c.name);
    }
    
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
  } finally {
    if (client) await client.close();
  }
});

export default router; 