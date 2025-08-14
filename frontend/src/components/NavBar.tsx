import React, { useState, FormEvent, useEffect } from 'react';
import successLogo from '../Assets/Success.jpg'; // adjust the path
import '../css/NavBar.css';
import { AuthServer } from '../clients/AuthServer';
import Stock from './Stock';
import News from './News';
import Portfolio from './Portfolio';

const authServer = new AuthServer('https://localhost:55604');

const NavBar: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchTermFinal, setSearchTermFinal] = useState('');
  const [currentPage, setCurrentPage] = useState('');
  const [showAccountModal, setShowAccountModal] = useState(false);
  const [showUserProfileModal, setShowUserProfileModal] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [username, setUsername] = useState('');

  const [accountUsername, setAccountUsername] = useState('');
  const [accountPassword, setAccountPassword] = useState('');
  const [loginError, setLoginError] = useState('');

  // Check if user is already logged in on component mount
  useEffect(() => {
    if (AuthServer.isAuthenticated()) {
      setIsLoggedIn(true);
      const savedUsername = AuthServer.getUsername();
      if (savedUsername) {
        setUsername(savedUsername);
      }
    }
  }, []);

  const handleSearch = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSearchTermFinal(searchTerm)
    console.log('Searching for:', searchTerm);
    // Your search logic here
  };

  const toggleAccountModal = () => {
    setShowAccountModal(!showAccountModal);
  };

  const toggleUserProfileModal = () => {
    setShowUserProfileModal(!showUserProfileModal);
  };

  const handleAccountLogin = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      const loginResponse = await authServer.login(accountUsername, accountPassword);
      console.log('Login successful, token:', loginResponse);
      
      setShowAccountModal(false);
      setIsLoggedIn(true);
      setUsername(accountUsername);
      setSearchTerm("AAPL")
      setSearchTermFinal("AAPL")
    } catch (error: Error | unknown) {
      console.error('Login error:', error);
      setLoginError(error instanceof Error ? error.message : 'Login failed');
    }
  };

  const handleLogout = () => {
    // Use the AuthServer to handle logout
    AuthServer.logout();
    
    // Update component state
    setIsLoggedIn(false);
    setUsername('');
    setShowUserProfileModal(false);
  };

  return (
    <>
      <nav className="navbar">
        {/* Left Section: Logo */}
        <div className="navbar__left">
          <img src={successLogo} alt="Success Logo" className="navbar__logo" />
        </div>

        <div className="navbar__left">
            <button onClick={() => setCurrentPage("portfolio")} className="search-button">
              Portfolio
            </button>
        </div>

        <div className="navbar__left">
            <button onClick={() => setCurrentPage("market")} className="search-button px-5">
              Market
            </button>
        </div>

        {/* Center Section: Search Bar */}
        <div className="navbar__center">
          <form onSubmit={handleSearch} className="search-form">
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search..."
              className="search-input"
            />
            <button type="submit" className="search-button">
              Search
            </button>
          </form>
        </div>

        {/* Right Section: Navigation Links */}
        <div className="navbar__right">
          <a href="#home" className="nav-link">
            Home
          </a>
          {isLoggedIn ? (
            <div className="user-profile">
              <span 
                className="username" 
                onClick={toggleUserProfileModal}
                style={{ cursor: 'pointer' }}
              >
                Welcome, {username}
              </span>
            </div>
          ) : (
            <button
              type="button"
              onClick={toggleAccountModal}
              className="nav-link button-link"
            >
              Login
            </button>
          )}
        </div>
      </nav>
      {
        currentPage == "market" ? (
          <Stock searchTerm={searchTermFinal} isLoggedIn={isLoggedIn} />
        ) : (
          <div></div>
        )
      }
      {
        currentPage == "market" ? (
          <News searchTerm={searchTermFinal} isLoggedIn={isLoggedIn}/>
        ) : (
          <div></div>
        )
      }
      {
        currentPage == "portfolio" ? (
          <Portfolio setSearchTermFinal={setSearchTermFinal} setSearchTerm={setSearchTerm} setCurrentPage={setCurrentPage} searchTerm={searchTermFinal} isLoggedIn={isLoggedIn} />
        ) : (
          <div></div>
        )
      }

      {/* Account Modal */}
      {showAccountModal && !isLoggedIn && (
        <div className="modal-overlay" onClick={toggleAccountModal}>
          <div
            className="modal-content"
            onClick={(e) => e.stopPropagation()} // Prevent modal close when clicking inside
          >
            <button className="modal-close" onClick={toggleAccountModal}>
              &times;
            </button>
            <h2>Account Login</h2>
            {loginError && <p className="error">{loginError}</p>}
            <form onSubmit={handleAccountLogin}>
              <div className="form-row">
                <label htmlFor="account-username">Username:</label>
                <input
                  id="account-username"
                  type="text"
                  value={accountUsername}
                  onChange={(e) => setAccountUsername(e.target.value)}
                />
              </div>
              <div className="form-row">
                <label htmlFor="account-password">Password:</label>
                <input
                  id="account-password"
                  type="password"
                  value={accountPassword}
                  onChange={(e) => setAccountPassword(e.target.value)}
                />
              </div>
              <button type="submit">Login</button>
            </form>
          </div>
        </div>
      )}  

      {/* User Profile Modal */}
      {showUserProfileModal && isLoggedIn && (
        <div className="modal-overlay" onClick={toggleUserProfileModal}>
          <div
            className="modal-content user-profile-modal"
            onClick={(e) => e.stopPropagation()} // Prevent modal close when clicking inside
          >
            <button className="modal-close" onClick={toggleUserProfileModal}>
              &times;
            </button>
            <h2>User Profile</h2>
            
            <div className="user-info">
              <div className="user-info-row">
                <span className="info-label">Username:</span>
                <span className="info-value">{username}</span>
              </div>
              <div className="user-info-row">
                <span className="info-label">Status:</span>
                <span className="info-value">Logged In</span>
              </div>
              <div className="user-info-row">
                <span className="info-label">Session:</span>
                <span className="info-value">Active</span>
              </div>
            </div>

            <div className="user-actions">
              <button 
                className="user-action-button profile-button"
                onClick={() => {
                  toggleUserProfileModal();
                  // Navigate to profile page if needed
                }}
              >
                View Full Profile
              </button>
              <button 
                className="user-action-button logout-button"
                onClick={handleLogout}
              >
                Log Out
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default NavBar;