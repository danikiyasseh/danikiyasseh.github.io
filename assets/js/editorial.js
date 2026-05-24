(function () {
	'use strict';

	document.getElementById('mobile-menu-btn')?.addEventListener('click', function () {
		document.getElementById('mobile-menu')?.classList.toggle('open');
	});

	(function () {
		var els = document.querySelectorAll('.ed-reveal');
		if (!('IntersectionObserver' in window)) {
			els.forEach(function (el) { el.classList.add('in'); });
			return;
		}
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (entry) {
				if (entry.isIntersecting) {
					entry.target.classList.add('in');
					io.unobserve(entry.target);
				}
			});
		}, { threshold: 0.15, rootMargin: '0px 0px -8% 0px' });
		els.forEach(function (el) { io.observe(el); });
	})();

	(function () {
		var targets = document.querySelectorAll('[data-count]');
		function animate(el) {
			var target = parseFloat(el.dataset.count);
			var format = el.dataset.format;
			var dur = 1600;
			var start = performance.now();
			function ease(t) { return 1 - Math.pow(1 - t, 3); }
			function step(now) {
				var t = Math.min(1, (now - start) / dur);
				var v = target * ease(t);
				el.textContent = format === 'comma'
					? Math.round(v).toLocaleString('en-US')
					: Math.round(v).toString();
				if (t < 1) requestAnimationFrame(step);
			}
			requestAnimationFrame(step);
		}
		if (!('IntersectionObserver' in window)) {
			targets.forEach(animate);
			return;
		}
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (entry) {
				if (entry.isIntersecting) {
					animate(entry.target);
					io.unobserve(entry.target);
				}
			});
		}, { threshold: 0.5 });
		targets.forEach(function (t) { io.observe(t); });
	})();
})();
