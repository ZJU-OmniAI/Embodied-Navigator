(() => {
  const dialog = document.querySelector("[data-lightbox-dialog]");
  const dialogImage = document.querySelector("[data-lightbox-image]");
  const dialogPlayer = document.querySelector("[data-lightbox-player]");
  const dialogCaption = document.querySelector("[data-lightbox-caption]");
  const closeButton = document.querySelector("[data-lightbox-close]");
  const figureButtons = document.querySelectorAll("[data-lightbox]");
  const videoButtons = document.querySelectorAll("[data-lightbox-video]");
  let returnFocus = null;

  const closeLightbox = () => {
    if (!dialog || dialog.hidden) {
      return;
    }

    dialog.hidden = true;
    document.body.classList.remove("lightbox-open");
    dialogImage.removeAttribute("src");
    dialogImage.alt = "";
    dialogImage.hidden = false;
    if (dialogPlayer) {
      dialogPlayer.pause();
      dialogPlayer.removeAttribute("src");
      dialogPlayer.load();
      dialogPlayer.hidden = true;
    }
    dialogCaption.textContent = "";

    if (returnFocus) {
      returnFocus.focus();
      returnFocus = null;
    }
  };

  const openLightbox = (button) => {
    if (!dialog) {
      return;
    }

    const sourceImage = button.querySelector("img");
    returnFocus = button;
    dialogImage.src = button.dataset.lightbox;
    dialogImage.alt = sourceImage ? sourceImage.alt : "Expanded research figure";
    dialogImage.hidden = false;
    if (dialogPlayer) {
      dialogPlayer.hidden = true;
    }
    dialogCaption.textContent = button.dataset.caption || "";
    dialog.hidden = false;
    document.body.classList.add("lightbox-open");
    closeButton.focus();
  };

  const openVideoLightbox = (button) => {
    if (!dialog || !dialogPlayer) {
      return;
    }

    returnFocus = button;
    dialogImage.removeAttribute("src");
    dialogImage.alt = "";
    dialogImage.hidden = true;
    dialogPlayer.src = button.dataset.lightboxVideo;
    dialogPlayer.hidden = false;
    dialogCaption.textContent = button.dataset.caption || "";
    dialog.hidden = false;
    document.body.classList.add("lightbox-open");
    dialogPlayer.play().catch(() => {});
    dialogPlayer.focus();
  };

  figureButtons.forEach((button) => {
    button.addEventListener("click", () => openLightbox(button));
  });

  videoButtons.forEach((button) => {
    button.addEventListener("click", () => openVideoLightbox(button));
  });

  if (closeButton) {
    closeButton.addEventListener("click", closeLightbox);
  }

  if (dialog) {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) {
        closeLightbox();
      }
    });
  }

  document.addEventListener("keydown", (event) => {
    if (!dialog || dialog.hidden) {
      return;
    }

    if (event.key === "Escape") {
      closeLightbox();
    }

    if (event.key === "Tab") {
      const playerVisible = dialogPlayer && !dialogPlayer.hidden;
      if (!playerVisible) {
        event.preventDefault();
        closeButton.focus();
        return;
      }
      // Cycle focus between the close button and the video player controls.
      event.preventDefault();
      if (document.activeElement === dialogPlayer) {
        closeButton.focus();
      } else {
        dialogPlayer.focus();
      }
    }
  });

  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    const source = document.getElementById(button.dataset.copyTarget);
    if (!source) {
      return;
    }

    let resetTimer = null;
    button.addEventListener("click", async () => {
      const text = source.textContent.trim();
      let copied = false;

      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch {
        const range = document.createRange();
        range.selectNodeContents(source);
        const selection = window.getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        copied = document.execCommand("copy");
        selection.removeAllRanges();
      }

      button.textContent = copied ? "Copied" : "Copy failed";
      button.classList.toggle("is-copied", copied);
      window.clearTimeout(resetTimer);
      resetTimer = window.setTimeout(() => {
        button.textContent = "Copy";
        button.classList.remove("is-copied");
      }, 2000);
    });
  });

  const navLinks = Array.from(document.querySelectorAll(".nav-links a"));
  const sections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if (sections.length) {
    const setActiveLink = (id) => {
      navLinks.forEach((link) => {
        const active = link.getAttribute("href") === "#" + id;
        link.classList.toggle("is-active", active);
        if (active) {
          link.setAttribute("aria-current", "location");
        } else {
          link.removeAttribute("aria-current");
        }
      });
    };

    let navFrame = null;
    const updateActiveLink = () => {
      const headerHeight = document.querySelector("[data-site-header]")?.offsetHeight || 0;
      const readingLine = window.scrollY + headerHeight + window.innerHeight * 0.22;
      let activeId = "";

      sections.forEach((section) => {
        if (section.offsetTop <= readingLine) {
          activeId = section.id;
        }
      });

      setActiveLink(activeId);
      navFrame = null;
    };

    const scheduleNavUpdate = () => {
      if (navFrame === null) {
        navFrame = window.requestAnimationFrame(updateActiveLink);
      }
    };

    window.addEventListener("scroll", scheduleNavUpdate, { passive: true });
    window.addEventListener("resize", scheduleNavUpdate);
    window.addEventListener("hashchange", scheduleNavUpdate);
    updateActiveLink();
  }
})();
