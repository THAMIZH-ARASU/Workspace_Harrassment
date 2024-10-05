import React from "react";
import { BrowserRouter as Router, Route, Routes, Link } from "react-router-dom";
import Home from "./Home";
import Dashboard from "./Dashboard";

function App() {
  return (
    <Router>
      <div>
        <header>
          <div>
            <h1 className="title">
              <span className="PinPep">P</span>epper
            </h1>
          </div>
          <nav className="nav_part">
            <Link to="/" className="head_nav">
              Home
            </Link>
            <Link to="/dashboard" className="head_nav">
              Dashboard
            </Link>
            <Link to="/footage" className="head_nav">
              Real-Time Footage
            </Link>
            <Link to="/contributors" className="head_nav">
              Contributors
            </Link>
            <Link to="/learn-more" className="head_nav">
              Learn More
            </Link>
          </nav>
        </header>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          {/* Add other routes here */}
        </Routes>
      </div>
      <footer>
        <p>&copy; 2024 Pepper. All rights reserved.</p>
      </footer>
    </Router>
  );
}

export default App;
