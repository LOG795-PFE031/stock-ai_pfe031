import forge from 'node-forge';
import axios from 'axios';
import { chatbotService } from './ChatbotService';

export class AuthServer {
    private baseUrl: string;
    private static readonly TOKEN_KEY = 'authToken';
    private static readonly USERNAME_KEY = 'username';

    constructor(baseUrl: string) {
      this.baseUrl = baseUrl;
    }

    /**
     * Get the user wallet ID
     */
    public async getUserWalletId() 
    {
        const username = AuthServer.getUsername()
        const walletResponse = await axios.get(`${this.baseUrl}/user/wallet`, {
            params:{
                username : username
            }
        })
        return walletResponse
    }

    /**
     * Get the user wallet
     */
    public async getUserShares(userWalletId : string) 
    {
        console.log(userWalletId)
        const shareResponse = await axios.get(`https://localhost:55616/portfolio/${userWalletId}`)
        return shareResponse
    }

    public async createUser (
        username : string, 
        password : string
    ){
        try{
            const role = 'Client'
            const requestBody = {
                username,
                password,
                role 
            };
            const response = await fetch(`${this.baseUrl}/users`,{
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            })

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error response:', errorText);
                throw new Error(`Sign up failed with status: ${response.status} - ${errorText}`);
            }
        }
        catch(error){
            console.log(error)
            throw(error)
        }
        
    }
  
    /**
     * Sends a POST request to the /login endpoint with username/password.
     * Returns a Promise with the parsed JSON data on success.
     */
    public async login(
      username: string,
      password: string
    ): Promise<string> {
        try {
            console.log(`Connecting to auth server at: ${this.baseUrl}`);
            
            // Create the expected format - simple username and password
            const requestBody = {
                username,
                password
            };
            
            console.log('Sending signin request with credentials');
            
            const response = await fetch(`${this.baseUrl}/auth/signin`, {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
            });
        
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error response:', errorText);
                throw new Error(`Login failed with status: ${response.status} - ${errorText}`);
            }
        
            const token = await response.text();
            console.log('Got token, length:', token.length);

            // Save token to localStorage and set in axios
            AuthServer.setAuthToken(token);
            AuthServer.setUsername(username);

            console.log('Validating token...');
            const validationResponse = await fetch(`${this.baseUrl}/user/validate`, {
                method: 'GET',
                headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
                }
            });

            if (!validationResponse.ok) {
                const errorText = await validationResponse.text();
                throw new Error(`Login Validation failed with status: ${validationResponse.status} - ${errorText}`);
            }

            console.log('Token validated successfully');
            
            // Initialize or restore chat session with ChatbotService
            try {
                await chatbotService.handleLogin(username, token);
                console.log('Chat session initialized for user:', username);
            } catch (error) {
                console.error('Failed to initialize chat session:', error);
                // Continue with login even if chat session fails
            }
            
            return token;
        } catch (error) {
            console.error('AuthServer login error:', error);
            throw error;
        }
    }

    /**
     * Alternative login method using encrypted credentials.
     * This can be used if the first approach doesn't work.
     */
    public async loginEncrypted(
        username: string,
        password: string
    ): Promise<string> {
        try {
            console.log(`[Encrypted] Connecting to auth server at: ${this.baseUrl}`);
            
            const publicKeyResponse = await fetch(`${this.baseUrl}/auth/publickey`, {
                method: 'GET',
                headers: {
                  'Content-Type': 'application/json',
                },
            });
              
            if (!publicKeyResponse.ok) {
                throw new Error(`Failed to get public key: ${publicKeyResponse.status} - ${await publicKeyResponse.text()}`);
            }

            const publicKeyData: PublicKeyResponse = await publicKeyResponse.json();
            console.log('[Encrypted] Got public key');

            const credentials: LoginRequest = { username, password };
            
            // Encrypt the data
            const encryptedData = encryptCredentials(publicKeyData.publicKey, credentials);
            
            // Try with encrypted data format
            const response = await fetch(`${this.baseUrl}/auth/signin`, {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    username: encryptedData,
                    password: encryptedData
                }),
            });
        
            if (!response.ok) {
                const errorText = await response.text();
                console.error('[Encrypted] Server error response:', errorText);
                throw new Error(`Login failed with status: ${response.status} - ${errorText}`);
            }
        
            const token = await response.text();
            console.log('[Encrypted] Got token');

            // Save token to localStorage and set in axios
            AuthServer.setAuthToken(token);
            AuthServer.setUsername(username);

            const validationResponse = await fetch(`${this.baseUrl}/user/validate`, {
                method: 'GET',
                headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
                }
            });

            if (!validationResponse.ok) {
                const errorText = await validationResponse.text();
                throw new Error(`Login Validation failed with status: ${validationResponse.status} - ${errorText}`);
            }

            return token;
        } catch (error) {
            console.error('[Encrypted] AuthServer login error:', error);
            throw error;
        }
    }

    /**
     * Set the authentication token and configure axios with it
     */
    public static setAuthToken(token: string): void {
        // Save token to localStorage
        localStorage.setItem(this.TOKEN_KEY, token);
        
        // Set token in axios default headers for all future requests
        axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
    }

    /**
     * Set the username in localStorage
     */
    public static setUsername(username: string): void {
        localStorage.setItem(this.USERNAME_KEY, username);
    }

    /**
     * Get the current authentication token
     */
    public static getAuthToken(): string | null {
        return localStorage.getItem(this.TOKEN_KEY);
    }

    /**
     * Get the current username
     */
    public static getUsername(): string | null {
        return localStorage.getItem(this.USERNAME_KEY);
    }

    /**
     * Check if user is authenticated (has a token)
     */
    public static isAuthenticated(): boolean {
        return !!this.getAuthToken();
    }

    /**
     * Remove the auth token and clear axios headers
     */
    public static logout(): void {
        const username = this.getUsername();
        if (username) {
            // Save chat history before logging out
            try {
                chatbotService.handleLogout().then(() => {
                    console.log('Chat session ended and saved for user:', username);
                }).catch(error => {
                    console.error('Error ending chat session:', error);
                });
            } catch (error) {
                console.error('Failed to end chat session:', error);
            }
        }
        
        localStorage.removeItem(this.TOKEN_KEY);
        localStorage.removeItem(this.USERNAME_KEY);
        delete axios.defaults.headers.common['Authorization'];
    }

    /**
     * Initialize the auth state from localStorage (call this when the app starts)
     */
    public static initializeAuth(): void {
        const token = this.getAuthToken();
        if (token) {
            axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        }
    }

    /**
     * Decode and parse the JWT token to get user information
     */
    public static decodeToken(): Record<string, unknown> | null {
        const token = this.getAuthToken();
        if (!token) return null;

        try {
            // Split the token and get the payload section
            const base64Url = token.split('.')[1];
            const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
            
            // Decode the payload
            const jsonPayload = decodeURIComponent(
                atob(base64)
                    .split('')
                    .map(c => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                    .join('')
            );

            return JSON.parse(jsonPayload);
        } catch (e) {
            console.error('Error decoding token:', e);
            return null;
        }
    }
}

export const encryptCredentials = (publicKeyPem: string, credentials: LoginRequest): string => {
  try {
    // Make sure the public key is properly formatted
    if (!publicKeyPem.includes('-----BEGIN')) {
      throw new Error('Invalid public key format');
    }
    
    // Parse the public key
    const publicKey = forge.pki.publicKeyFromPem(publicKeyPem);
    
    // Stringify the credentials with proper JSON
    const credentialsJson = JSON.stringify(credentials);
    console.log('Credentials JSON length:', credentialsJson.length);
    
    // Encrypt with RSA-OAEP
    const encryptedData = publicKey.encrypt(credentialsJson, 'RSA-OAEP', {
      md: forge.md.sha256.create(),
    });
    
    // Base64 encode the result
    const base64Result = forge.util.encode64(encryptedData);
    console.log('Encrypted data length (base64):', base64Result.length);
    
    return base64Result;
  } catch (error) {
    console.error('Encryption error:', error);
    throw new Error(`Failed to encrypt credentials: ${error instanceof Error ? error.message : String(error)}`);
  }
};

export interface PublicKeyResponse {
    publicKey: string;
  }

export interface LoginRequest {
    username: string;
    password: string;
  }