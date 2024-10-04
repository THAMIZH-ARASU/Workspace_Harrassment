document.addEventListener("DOMContentLoaded", () => {
  const realTimeBtn = document.getElementById("real-time-btn");
  const dashboardBtn = document.getElementById("dashboard-btn");
  const infoBoxes = document.querySelectorAll(".info-box");

  // Function to show a specific info box
  const showInfoBox = (indexToShow) => {
    infoBoxes.forEach((box, index) => {
      if (index === indexToShow) {
        box.style.opacity = "1";
      } else {
        box.style.opacity = "0";
      }
    });
  };

  // Default: show info-box-0
  showInfoBox(0);

  // Add event listeners for buttons
  realTimeBtn.addEventListener("mouseenter", () => {
    showInfoBox(1); // Show info-box-1 when hovering over Real-Time button
  });

  dashboardBtn.addEventListener("mouseenter", () => {
    showInfoBox(2); // Show info-box-2 when hovering over Dashboard button
  });

  // When the mouse leaves both buttons, revert to info-box-0
  realTimeBtn.addEventListener("mouseleave", () => {
    showInfoBox(0);
  });

  dashboardBtn.addEventListener("mouseleave", () => {
    showInfoBox(0);
  });
});

// JavaScript to handle the hover effect for buttons
const buttons = document.querySelectorAll(".click-btn");
const backgroundP = document.querySelector(".background-p");

buttons.forEach((button) => {
  button.addEventListener("mouseenter", () => {
    backgroundP.style.opacity = 0;
  });

  button.addEventListener("mouseleave", () => {
    backgroundP.style.opacity = 1;
  });
});
