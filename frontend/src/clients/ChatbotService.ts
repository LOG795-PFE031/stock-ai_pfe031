import { ChatOpenAI } from "@langchain/openai";
import { ConversationSummaryBufferMemory, ConversationSummaryBufferMemoryInput } from "langchain/memory";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { LLMChain, ConversationChain } from "langchain/chains";
import { PromptTemplate } from "@langchain/core/prompts";

// Risk tolerance options
export const RISK_TOLERANCE_OPTIONS = {
  '1': 'low',
  '2': 'medium',
  '3': 'high'
};

// Investment goals options
export const INVESTMENT_GOALS_OPTIONS = {
  '1': 'Conservative Income (Focus on stable, dividend-paying investments)',
  '2': 'Balanced Growth (Mix of growth and income)',
  '3': 'Aggressive Growth (Focus on capital appreciation)',
  '4': 'Retirement Planning',
  '5': 'Short-term Trading'
};

// Available sectors
export const AVAILABLE_SECTORS = {
  '1': 'Technology',
  '2': 'Healthcare',
  '3': 'Financial Services',
  '4': 'Consumer Goods',
  '5': 'Energy',
  '6': 'Real Estate',
  '7': 'Industrial',
  '8': 'Communications',
  '9': 'Materials',
  '10': 'Utilities'
};

// User profile interface
interface UserProfile {
  userId: string;
  riskTolerance: string;
  investmentGoals: string;
  preferredSectors: string[];
}

// Conversation interfaces
interface ProfileUpdate {
  risk_tolerance?: string;
  investment_goals?: string;
  preferred_sectors?: string[];
}

// Function definition for updating profile
const updateProfileFunction = {
  name: "update_profile",
  description: "Update the user's financial profile based on their input.",
  parameters: {
    type: "object",
    properties: {
      risk_tolerance: {
        type: "string",
        enum: ["low", "medium", "high"],
        description: "User's risk tolerance level"
      },
      investment_goals: {
        type: "string",
        description: "User's investment goals (e.g., growth, income)"
      },
      preferred_sectors: {
        type: "array",
        items: { type: "string" },
        description: "Sectors the user is interested in (e.g., tech, healthcare)"
      }
    },
    required: []
  }
};

// Simple memory class that extends ConversationSummaryBufferMemory
export class SimpleConversationMemory extends ConversationSummaryBufferMemory {
  constructor(config: ConversationSummaryBufferMemoryInput) {
    super(config);
    console.log("SimpleConversationMemory initialized");
  }
}

export class ChatbotService {
  private model!: ChatOpenAI;
  private memory!: SimpleConversationMemory | null;
  private apiKey: string;
  private userId: string;
  private userProfile: UserProfile = {
    userId: "user1",
    riskTolerance: "medium",
    investmentGoals: "Balanced Growth (Mix of growth and income)",
    preferredSectors: ["Technology", "Healthcare", "Financial Services"]
  };

  constructor(userId: string = "user1") {
    console.log("ChatbotService constructor called");
    this.apiKey = import.meta.env.VITE_OPENAI_API_KEY || '';
    
    // Try to restore session from localStorage
    const savedUserId = localStorage.getItem('chatbot_user_id');
    const lastAuth = localStorage.getItem('chatbot_last_auth');
    
    // Use saved userId if available and not older than 24 hours
    if (savedUserId && lastAuth) {
      const lastAuthTime = new Date(lastAuth).getTime();
      const currentTime = new Date().getTime();
      const hoursSinceLastAuth = (currentTime - lastAuthTime) / (1000 * 60 * 60);
      
      if (hoursSinceLastAuth < 24) {
        console.log(`Restoring session for user ${savedUserId}`);
        this.userId = savedUserId;
      } else {
        console.log('Previous session expired, using provided userId');
        this.userId = userId;
      }
    } else {
      console.log('No previous session found, using provided userId');
      this.userId = userId;
    }
    
    console.log(`ChatbotService initialized with userId: ${this.userId}`);
    this.initializeModel();
    
    // Initialize for current user
    this.setUserId(this.userId);
  }

  /** Initialize the OpenAI model */
  private initializeModel() {
    try {
      this.model = new ChatOpenAI({
        openAIApiKey: this.apiKey,
        modelName: "gpt-4.1",
        temperature: 0.7,
        streaming: true,
      });
    } catch (error) {
      console.error("Error initializing the chatbot model:", error);
    }
  }

  /** Initialize memory for the current user */
  private async initializeMemory() {
    if (this.userId) {
      console.log(`Initializing memory for user: ${this.userId}`);
      
      // Make sure model is initialized with API key
      if (!this.model) {
        this.initializeModel();
      }
      
      // Create a new memory instance with this user's ID
      this.memory = new SimpleConversationMemory({
        llm: this.model,
        maxTokenLimit: 2000,
        returnMessages: true,
        memoryKey: "history",
      });
      
      console.log("Memory initialized successfully");
    } else {
      console.warn("No userId provided, using default memory");
    }
  }

  /** Send a message to the chatbot with streaming support */
  async sendMessage(message: string, onTokenStream?: (token: string) => void): Promise<string> {
    try {
      if (!this.model) {
        this.initializeModel();
      }
      
      // Ensure memory is initialized
      if (!this.memory) {
        await this.initializeMemory();
      }

      // Return error if memory is still null
      if (!this.memory) {
        console.error("Failed to initialize memory");
        return "Sorry, I'm having trouble accessing your conversation history. Please try again later.";
      }

      // First save the user's message to memory
      await this.memory.saveContext({ input: message }, { output: "" });
      
      // Load memory including previous conversation history
      const memoryVariables = await this.memory.loadMemoryVariables({});
      const memoryMessages = memoryVariables.history || [];

      // Extract previous conversation summary if available
      let summaryContent = "";
      const conversationHistory = [];
      
      for (const msg of memoryMessages) {
        if (msg._getType && typeof msg._getType === 'function') {
          const type = msg._getType();
          if (type === "system") {
            summaryContent = msg.content;
          } else if (type === "human") {
            conversationHistory.push({ role: "user", content: msg.content });
          } else if (type === "ai") {
            conversationHistory.push({ role: "assistant", content: msg.content });
          }
        }
      }

      // Create a detailed system message that includes all user profile information
      const systemContent = `You are a financial advisor specializing in stocks. 
The user has the following established profile:
- Risk Tolerance: ${this.userProfile.riskTolerance}
- Investment Goals: ${this.userProfile.investmentGoals}
- Preferred Sectors: ${this.userProfile.preferredSectors.join(', ')}

Use this profile information to provide personalized advice. 
Always reference specific details from past conversations, including exact investment amounts and stocks mentioned. 
Be precise when recalling past information.
Previous conversation summary: ${summaryContent}`;

      const messages = [
        { role: "system", content: systemContent },
        ...conversationHistory,
        { role: "user", content: message }
      ];

      let fullResponse = "";
      if (onTokenStream) {
        const streamingResponse = await this.model.invoke(messages, {
          tools: [{ type: "function", function: updateProfileFunction }],
          callbacks: [{
            handleLLMNewToken(token: string) {
              fullResponse += token;
              onTokenStream(token);
            },
          }],
        });

        // Explicitly save the bot's response to memory
        if (this.memory) {
          await this.memory.saveContext({ input: message }, { output: fullResponse });
        }

        if (streamingResponse.additional_kwargs?.tool_calls) {
          const toolCall = streamingResponse.additional_kwargs.tool_calls[0];
          const functionName = toolCall.function?.name;
          const functionArgs = JSON.parse(toolCall.function?.arguments || "{}");

          if (functionName === "update_profile") {
            this.updateProfile(functionArgs);
            const updateMessage = "\n\nI've updated your financial profile based on this information.";
            onTokenStream(updateMessage);
            fullResponse += updateMessage;
            // Save the updated response
            if (this.memory) {
              await this.memory.saveContext({ input: message }, { output: fullResponse });
            }
          }
        }
        return fullResponse;
      } else {
        const response = await this.model.invoke(messages, {
          tools: [{ type: "function", function: updateProfileFunction }]
        });

        if (response.additional_kwargs?.tool_calls) {
          const toolCall = response.additional_kwargs.tool_calls[0];
          const functionName = toolCall.function?.name;
          const functionArgs = JSON.parse(toolCall.function?.arguments || "{}");

          if (functionName === "update_profile") {
            this.updateProfile(functionArgs);
            const secondMessages = [
              { role: "system", content: systemContent },
              ...conversationHistory,
              { role: "assistant", content: "Profile updated successfully. " + response.content }
            ];
            const secondResponse = await this.model.invoke(secondMessages);
            const finalMessage = String(secondResponse.content);
            // Ensure message is saved to memory
            if (this.memory) {
              await this.memory.saveContext({ input: message }, { output: finalMessage });
            }
            return finalMessage;
          }
        }

        const finalMessage = String(response.content);
        // Ensure message is saved to memory
        if (this.memory) {
          await this.memory.saveContext({ input: message }, { output: finalMessage });
        }
        return finalMessage;
      }
    } catch (error) {
      console.error("Error sending message to chatbot:", error);
      return "Sorry, I'm having trouble processing your request. Please try again later.";
    }
  }

  /** Set the user ID and initialize user-specific data */
  async setUserId(userId: string) {
    try {
      console.log(`Setting user ID from ${this.userId} to ${userId}`);
      
      if (this.userId !== userId) {
        // User has changed, we need to reinitialize everything
        this.userId = userId;
        
        // Store user ID in localStorage for session persistence
        localStorage.setItem('chatbot_user_id', userId);
        
        // Save current time as last authentication time
        localStorage.setItem('chatbot_last_auth', new Date().toISOString());
        
        console.log(`Reinitializing memory and profile for user ${userId}`);
        
        // Then initialize memory with conversation history
        await this.initializeMemory();
        
        console.log(`Successfully reinitialized for user ${userId}`);
        return true;
      } else {
        // Same user, just refresh data
        console.log(`Refreshed data for existing user ${userId}`);
        return true;
      }
    } catch (error) {
      console.error(`Error setting user ID to ${userId}:`, error);
      return false;
    }
  }

  /** Set the OpenAI API key */
  setApiKey(apiKey: string) {
    this.apiKey = apiKey;
    this.initializeModel();
  }

  /** Get the user's current profile */
  getUserProfile(): UserProfile {
    return { ...this.userProfile };
  }

  /** Start the profile setup process */
  async startProfileSetup(riskTolerance: string, investmentGoals: string, preferredSectors: string[]) {
    this.userProfile = { userId: this.userId, riskTolerance, investmentGoals, preferredSectors };
    return this.userProfile;
  }
  
  /** End conversation and save summary */
  async endConversation(): Promise<string> {
    try {
      if (!this.memory) {
        await this.initializeMemory();
      }
      
      // Create a summary of the conversation
      return "Conversation ended successfully";
    } catch (error) {
      console.error("Error ending conversation:", error);
      return "Error ending conversation";
    }
  }

  /** Handle user login event */
  async handleLogin(userId: string, authToken: string): Promise<boolean> {
    try {
      console.log(`User logged in: ${userId}`);
      
      // Set the user ID which will load memory and profile
      const success = await this.setUserId(userId);
      
      if (success) {
        console.log('Successfully restored user session after login');
        return true;
      } else {
        console.error('Failed to set userId after login');
        return false;
      }
    } catch (error) {
      console.error('Error handling login:', error);
      return false;
    }
  }

  /** Handle user logout event */
  async handleLogout(): Promise<boolean> {
    try {
      console.log('Ending and saving conversation before logout');
      
      // Get the current summary
      const summary = await this.endConversation();
      console.log('Conversation summary saved:', summary);
      
      return true;
    } catch (error) {
      console.error('Error handling logout:', error);
      return false;
    }
  }

  /** Update user profile and save changes */
  private updateProfile(updates: ProfileUpdate): boolean {
    try {
      if (updates.risk_tolerance) {
        this.userProfile.riskTolerance = updates.risk_tolerance;
      }
      if (updates.investment_goals) {
        this.userProfile.investmentGoals = updates.investment_goals;
      }
      if (updates.preferred_sectors && Array.isArray(updates.preferred_sectors)) {
        updates.preferred_sectors.forEach((sector: string) => {
          if (!this.userProfile.preferredSectors.includes(sector)) {
            this.userProfile.preferredSectors.push(sector);
          }
        });
      }
      return true;
    } catch (error) {
      console.error("Error updating user profile:", error);
      return false;
    }
  }

  /** Create a test message (simplified without MongoDB) */
  async createTestMessage(content: string = "Test message"): Promise<boolean> {
    try {
      console.log("Creating test message...");
      
      // Create a memory instance if needed
      if (!this.memory) {
        await this.initializeMemory();
      }
      
      if (!this.memory) {
        console.error("Failed to initialize memory");
        return false;
      }
      
      // Save a test message to memory (no MongoDB)
      await this.memory.saveContext(
        { input: "Test input" }, 
        { output: content }
      );
      
      console.log("Successfully saved test message to memory");
      return true;
    } catch (error) {
      console.error("Error creating test message:", error);
      return false;
    }
  }
}

// Create a single instance of the ChatbotService
export const chatbotService = new ChatbotService();
export default chatbotService;
