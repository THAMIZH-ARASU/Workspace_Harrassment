import React from "react";
import "./Home.css";
import { useState } from "react";
function Home() {
  const [opc1, setOpc1] = useState(0);
  const [opc2, setOpc2] = useState(0);
  const [opcp, setOpcp] = useState(1);
  const [shd, setShd] = useState(50);
  return (
    <div>
      <head>
        <link rel="stylesheet" href="home.css" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link
          href="https://fonts.googleapis.com/css2?family=Martian+Mono:wght@100;400;700&display=swap"
          rel="stylesheet"
        />

        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
        <link
          href="https://fonts.googleapis.com/css2?family=Lexend:wght@100..900&display=swap"
          rel="stylesheet"
        />
      </head>
      <main className="content">
        {/* Background P */}
        <div className="background-p" style={{ opacity: opcp }}>
          P
        </div>

        <div className="info-box" id="info-box-0">
          {/* <pre style={{ opacity: opcp }}>
            Sexual Harassment{"\n\t"} Detection System
          </pre> */}
        </div>

        <div
          className="info-box"
          id="info-box-1"
          style={{ marginBottom: "100px", opacity: opc1 }}
        >
          <img
            style={{
              maxWidth: "300px",
              maxHeight: "300px",
              marginLeft: "30px",
            }}
            src="https://www.pngall.com/wp-content/uploads/13/CCTV-Camera-PNG.png"
            alt="CCTV Icon"
          />
          <pre style={{ textShadow: `0 0 ${shd}px #e74c3c` }}>
            Harnessing the power of live footage{"\n"}for proactive monitoring.
          </pre>
        </div>

        <div className="info-box" id="info-box-2" style={{ opacity: opc2 }}>
          <img
            style={{ maxWidth: "350px", maxHeight: "250px" }}
            src="https://static.vecteezy.com/system/resources/thumbnails/008/329/474/small_2x/dashboard-icon-style-free-vector.jpg"
            alt="Dashboard Icon"
          />
          <pre style={{ textShadow: `0 0 ${shd}px #e74c3c` }}>
            Incident monitoring and{"\n\t"}
            quick action dashboard
          </pre>
        </div>

        <div className="buttons-container">
          <a href="#real_foot">
            <button
              className="click-btn"
              id="real-time-btn"
              onMouseEnter={() => {
                setOpc1(1);
                setOpcp(0);
                setShd(0);
              }}
              onMouseLeave={() => {
                setOpc1(0);
                setOpcp(1);
                setShd(50);
              }}
            >
              Real-Time Footage
            </button>
          </a>
          <a href="/dashboard">
            <button
              className="click-btn"
              id="dashboard-btn"
              style={{ padding: "20px 90px" }}
              onMouseEnter={() => {
                setOpc2(1);
                setOpcp(0);
                setShd(0);
              }}
              onMouseLeave={() => {
                setOpc2(0);
                setOpcp(1);
                setShd(50);
              }}
            >
              Dashboard
            </button>
          </a>
        </div>
      </main>
    </div>
  );
}

export default Home;
