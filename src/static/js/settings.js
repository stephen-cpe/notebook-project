/* notebook-project — avatar selection modal */
(function () {
  "use strict";

  var modal = document.getElementById("avatar-modal");
  var openBtn = document.getElementById("open-avatar-modal");
  var closeBtn = document.getElementById("close-avatar-modal");
  var confirmBtn = document.getElementById("confirm-avatar");
  var grid = document.getElementById("avatar-grid");
  var avatarInput = document.getElementById("avatar-input");
  var currentImg = document.getElementById("current-avatar-img");
  var form = document.getElementById("settings-form");

  var pendingAvatar = null;

  if (!openBtn || !modal) return;

  openBtn.addEventListener("click", function () {
    modal.hidden = false;
    pendingAvatar = avatarInput.value;
  });

  closeBtn.addEventListener("click", function () {
    modal.hidden = true;
  });

  modal.addEventListener("click", function (event) {
    if (event.target === modal) modal.hidden = true;
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape" && !modal.hidden) modal.hidden = true;
  });

  if (grid) {
    grid.addEventListener("click", function (event) {
      var cell = event.target.closest(".avatar-cell");
      if (!cell) return;
      grid.querySelectorAll(".avatar-cell").forEach(function (c) {
        c.classList.remove("is-selected");
      });
      cell.classList.add("is-selected");
      pendingAvatar = cell.dataset.avatar;
    });
  }

  if (confirmBtn) {
    confirmBtn.addEventListener("click", function () {
      if (pendingAvatar && pendingAvatar !== avatarInput.value) {
        avatarInput.value = pendingAvatar;
        if (currentImg) {
          currentImg.src = currentImg.src.replace(/[^/]*$/, pendingAvatar);
        }
      }
      modal.hidden = true;
      if (form) form.submit();
    });
  }
})();
