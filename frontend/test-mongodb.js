// MongoDB connection test and setup script
import { MongoClient } from 'mongodb';

async function testAndSetupMongoDB() {
  const url = 'mongodb://mongo:mongo@localhost:27017';
  console.log(`Testing connection to MongoDB at: ${url}`);
  
  try {
    const client = new MongoClient(url);
    await client.connect();
    console.log('✅ Successfully connected to MongoDB!');
    
    // Get available databases
    const databases = await client.db().admin().listDatabases();
    console.log('Available databases:');
    databases.databases.forEach(db => {
      console.log(`- ${db.name}`);
    });
    
    // Setup StockAI database and collections if they don't exist
    console.log("\nSetting up StockAI database and collections...");
    const db = client.db("StockAI");
    
    // Create collections
    const requiredCollections = ['conversations', 'summaries', 'users'];
    
    // Get list of existing collections
    const existingCollections = await db.listCollections().toArray();
    const existingCollectionNames = existingCollections.map(c => c.name);
    
    // Create missing collections
    for (const collName of requiredCollections) {
      if (!existingCollectionNames.includes(collName)) {
        console.log(`Creating collection: ${collName}`);
        await db.createCollection(collName);
        
        // Create test document in the collection
        if (collName === 'conversations') {
          await db.collection(collName).insertOne({
            userId: 'test-user',
            message: 'Test message',
            sender: 'system',
            timestamp: new Date()
          });
          console.log(`Added test document to ${collName} collection`);
        } else if (collName === 'summaries') {
          await db.collection(collName).insertOne({
            userId: 'test-user',
            summary: 'Test conversation summary',
            timestamp: new Date()
          });
          console.log(`Added test document to ${collName} collection`);
        } else if (collName === 'users') {
          await db.collection(collName).insertOne({
            userId: 'test-user',
            riskTolerance: 'medium',
            investmentGoals: 'Balanced Growth',
            preferredSectors: ['Technology'],
            timestamp: new Date()
          });
          console.log(`Added test document to ${collName} collection`);
        }
      } else {
        console.log(`Collection ${collName} already exists`);
      }
    }
    
    // List collections after setup
    const updatedCollections = await db.listCollections().toArray();
    console.log("\nCollections in StockAI database:");
    updatedCollections.forEach(c => {
      console.log(`- ${c.name}`);
    });
    
    await client.close();
    console.log('\nConnection closed');
    console.log('Database setup complete! ✅');
  } catch (err) {
    console.error('❌ Error:', err);
  }
}

testAndSetupMongoDB().catch(console.error); 