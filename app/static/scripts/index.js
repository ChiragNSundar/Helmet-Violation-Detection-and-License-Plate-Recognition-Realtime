/**
 * index.js
 * - All our useful JS goes here, awesome!
 */
/* Preloader */
document.onreadystatechange = function () {
    if (document.readyState !== "complete") {
      document.querySelector(".preloader-wrap").style.visibility = "visible";
    } else {
      document.querySelector(".preloader-wrap").style.display = "none";
    }
  };
  /* Initilize Aos Animation */
  
  AOS.init({
    duration: 1500
  });
  
  /* Cursor Animation */
  const mouseCursor = document.querySelector(".cursor");
  const navLinks = document.querySelectorAll("a");
  
  window.addEventListener("mousemove", cursorFunc);
  
  function cursorFunc(e) {
    mouseCursor.style.top = e.pageY + "px";
    mouseCursor.style.left = e.pageX + "px";
  }
  
  navLinks.forEach(function (link) {
    link.addEventListener("mouseout", function () {
      mouseCursor.classList.remove("link-grow");
      link.classList.remove("hovered-link");
    });
  
    link.addEventListener("mouseover", function () {
      mouseCursor.classList.add("link-grow");
      link.classList.add("hovered-link");
    });
  });
  
  /* Acordian Slider */
  $(".ourindustry-slider").slick({
    slidesToShow: 1,
    slidesToScroll: 1,
    arrows: true,
    fade: false,
    infinite: true,
    autoplay: false,
    asNavFor: ".ourindustry-accordion",
    draggable: false,
    prevArrow: '<i class="fa fa-long-arrow-left"></i>',
    nextArrow: '<i class="fa fa-long-arrow-right"></i>'
  });
  $(".ourindustry-accordion").slick({
    slidesToShow: 5,
    slidesToScroll: 1,
    asNavFor: ".ourindustry-slider",
    dots: false,
    //centerMode: true,
    focusOnSelect: true,
    vertical: true,
    verticalSwiping: true,
    draggable: false,
    adaptiveHeight: false
  });
  //remove active class from all left acordian slides
  $(".ourindustry-accordion .slick-slide").removeClass("slick-current");
  
  //set active class to first acrodian slide
  $(".ourindustry-accordion .slick-slide").eq(0).addClass("slick-current");
  
  // On before slide change match active thumbnail to current slide
  $(".ourindustry-slider").on(
    "beforeChange",
    function (event, slick, currentSlide, nextSlide) {
      var mySlideNumber = nextSlide;
      $(".ourindustry-accordion .slick-slide").removeClass("slick-current");
      $(".ourindustry-accordion .slick-slide")
        .eq(mySlideNumber)
        .addClass("slick-current");
    }
  );
  
  /* Main Slider */
  $(".main-slider").slick({
    autoplay: true,
    autoplaySpeed: 10000,
    speed: 1000,
    slidesToShow: 1,
    slidesToScroll: 1,
    pauseOnHover: false,
    dots: true,
    infinite: true,
    pauseOnDotsHover: true,
    cssEase: "linear",
    // fade:true,
    draggable: false,
    prevArrow: '<button class="PrevArrow"></button>',
    nextArrow: '<button class="NextArrow"></button>'
  });
  
  $(".main-slider").on("afterChange", function (event, slick, currentSlide) {
    var myvideo = $("#myvideo");
    if (currentSlide == 4) {
      $(".main-slider").slick("slickPause");
      myvideo.play();
    }
  });
  
  /* Contact Form */
  $(".contact-button").on("click", function () {
    var current = $(this);
    current.children("i").toggleClass("fa-envelope-o fa-times");
    current.toggleClass("contactopen");
    $(".main-wrapper").toggleClass("overflow-hidden");
    $(".contact-loader").addClass("open");
    setTimeout(function () {
      $(".contact-form").toggle();
      $(".contact-loader").removeClass("open");
       $(".main-wrapper").toggleClass("overflow-hidden");
    }, 2000);
  });
  
  /* right side text hide */
  
  $(window).on("load resize", function (e) {
    var windowwidth = $(window).width();
    if (windowwidth >= 1719) {
      $(".fixed-text").fadeIn(100);
    } else {
      $(".fixed-text").fadeOut();
    }
  });
  
  /* Header Fixed on scroll top */
  $(window).scroll(function () {
    var headersticky = $("header"),
      scrolltop = $(window).scrollTop(),
      headerheight = $("header").height();
    if (scrolltop >= 200) {
      headersticky.addClass("header-fixed");
      $(".main-wrapper").css("padding-top", headerheight);
    } else {
      headersticky.removeClass("header-fixed");
      $(".main-wrapper").css("padding-top", "0");
    }
  });
  
  /* Menu Toggle On responsive */
  $(".menu-toggle-button").on("click", function () {
    var current = $(this);
    current.toggleClass("on");
    $(".menu .main-list").toggleClass("main-list-open");
  });
  
  /* Sub menu */
  $(".dropmenu-btn").on("click", function (e) {
    e.preventDefault();
    var current = $(this);
    current.next().addClass("sub-list-open");
  });
  $(".submenu-backarrow").on("click", function () {
    var current = $(this);
    current.parents(".dropdown-menu").removeClass("sub-list-open");
  });
  
  /* light box */
  $(document).on("click", "[data-lightbox]", lity);
  
  /* Scroll TO respective Section  */
  $(".strip-buttons ul li a").click(function (e) {
    var current = $(this);
    e.preventDefault();
    var target = $(current.attr("href"));
    if (target.length) {
      var scrollTo = target.offset().top - 100;
      $("body, html").animate({ scrollTop: scrollTo + "px" }, 1500);
    }
  });
  
  /* Scroll Top */
  // declare variable
  var scrollTop = $(".scroll-top");
  scrollTop.css("opacity", "0");
  $(window).scroll(function () {
    var topPos = $(this).scrollTop();
    if (topPos > 100) {
      $(scrollTop).css("opacity", "1");
    } else {
      $(scrollTop).css("opacity", "0");
    }
  });
  //Click event to scroll to top
  $(scrollTop).click(function () {
    $("html, body").animate(
      {
        scrollTop: 0
      },
      1000
    );
    return false;
  });
  