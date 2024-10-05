import React from "react";
import "./Dashboard.css";

function Dashboard() {
  return (
    <div>
      {/* <head>
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
      </head> */}
      <main className="dashboard">
        {/* Overview Section */}
        <section className="overview">
          <div className="overview-box">
            <h2>Total Incidents</h2>
            <p>45</p>
          </div>
          <div className="overview-box">
            <h2>Active Alerts</h2>
            <p>5</p>
          </div>
        </section>

        {/* Recent Incidents List */}
        <section className="recent-incidents">
          <h2>Recent Incidents</h2>
          <table>
            <thead>
              <tr>
                <th>Date/Time</th>
                <th>Severity</th>
                <th>Location</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>2024-10-02 14:30</td>
                <td>High</td>
                <td>Office Entrance</td>
                <td>Unresolved</td>
              </tr>
              <tr>
                <td>2024-10-01 10:15</td>
                <td>Medium</td>
                <td>Parking Lot</td>
                <td>Resolved</td>
              </tr>
              {/* Add more rows as needed */}
            </tbody>
          </table>
        </section>

        {/* Real-Time Alerts Section */}
        <section className="real-time-alerts">
          <h2>Real-Time Alerts</h2>
          <p>No ongoing alerts</p>
        </section>

        {/* Actions Panel */}
        <section className="actions-panel">
          <h2>Quick Actions</h2>
          <button onClick={() => (window.location.href = "https://google.com")}>
            Resolve Incident
          </button>
          <button>Escalate Incident</button>
          <button>View Real-Time Footage</button>
        </section>

        {/* Summary Chart (Placeholder) */}
        <section className="summary-chart">
          <h2>Incident Trend (Last 7 Days)</h2>
          <div className="chart-placeholder">
            <p>[Simple Graph Placeholder]</p>
          </div>
        </section>

        {/* Settings/Notifications */}
        <section className="settings">
          <h2>Settings/Notifications</h2>
          <p>No new notifications</p>
        </section>
      </main>
    </div>
  );
}

export default Dashboard;
