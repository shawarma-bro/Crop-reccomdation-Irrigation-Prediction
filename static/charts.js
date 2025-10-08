// ==============================
// charts.js
// Enhancements for Crop Prediction UI
// ==============================

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const spinner = document.querySelector(".spinner");
  const resultCard = document.querySelector(".col-lg-7 .card");
  const clearBtn = form.querySelector("button[type='reset']");

  // Show spinner when form is submitted
  if (form) {
    form.addEventListener("submit", () => {
      spinner.style.display = "block";
      resultCard.classList.add("animate__fadeOut"); // fade old result
    });
  }

  // Clear form and results
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      setTimeout(() => {
        spinner.style.display = "none";
        resultCard.classList.remove("animate__fadeInRight", "animate__fadeOut");
        resultCard.querySelector(".card-body")?.classList.add("d-none");
      }, 150);
    });
  }

  // Animate results when page loads with prediction
  if (resultCard && resultCard.innerText.trim() !== "") {
    resultCard.classList.add("animate__animated", "animate__fadeInRight");
  }

  // Toastify flash messages if Bootstrap alerts exist
  const alerts = document.querySelectorAll(".alert");
  if (alerts.length > 0) {
    alerts.forEach(alert => {
      setTimeout(() => {
        alert.classList.add("animate__fadeOutUp");
        setTimeout(() => alert.remove(), 600);
      }, 4000); // auto-dismiss after 4s
    });
  }
});
