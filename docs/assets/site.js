(() => {
  const dialog = document.querySelector("[data-lightbox-dialog]");
  const dialogImage = document.querySelector("[data-lightbox-image]");
  const dialogCaption = document.querySelector("[data-lightbox-caption]");
  const closeButton = document.querySelector("[data-lightbox-close]");
  const figureButtons = document.querySelectorAll("[data-lightbox]");
  let returnFocus = null;

  const closeLightbox = () => {
    if (!dialog || dialog.hidden) {
      return;
    }

    dialog.hidden = true;
    document.body.classList.remove("lightbox-open");
    dialogImage.removeAttribute("src");
    dialogImage.alt = "";
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
    dialogCaption.textContent = button.dataset.caption || "";
    dialog.hidden = false;
    document.body.classList.add("lightbox-open");
    closeButton.focus();
  };

  figureButtons.forEach((button) => {
    button.addEventListener("click", () => openLightbox(button));
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
      event.preventDefault();
      closeButton.focus();
    }
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
