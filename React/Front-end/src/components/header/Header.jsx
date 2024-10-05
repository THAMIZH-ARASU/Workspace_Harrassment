import React from 'react'
import './home.css'
const Header = () => {
  return (
    <div>
      <header>
      <div>
        <h1 className="title">Pepper</h1>
      </div>

      <nav className="nav_part">
        <a href="home.html" className="head_nav">Home</a>
        <a href="dashboard.html" className="head_nav">Dashboard</a>
        <a href="footage.html" className="head_nav">Real-Time Footage</a>
        <a href="contributors.html" className="head_nav">Contributors</a>
        <a href="learn_more.html" className="head_nav">Learn More</a>
      </nav>
    </header>
    </div>
  )
}

export default Header
