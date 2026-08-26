(function () {
	'use strict';

	document.getElementById('mobile-menu-btn')?.addEventListener('click', function () {
		var menu = document.getElementById('mobile-menu');
		var nav = document.querySelector('.ed-nav');
		if (nav) {
			document.documentElement.style.setProperty('--ed-nav-height', nav.offsetHeight + 'px');
		}
		menu?.classList.toggle('open');
	});

	window.addEventListener('resize', function () {
		var nav = document.querySelector('.ed-nav');
		if (nav) {
			document.documentElement.style.setProperty('--ed-nav-height', nav.offsetHeight + 'px');
		}
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

	(function () {
		var root = document.getElementById('pub-filters');
		if (!root) return;

		var cards = Array.prototype.slice.call(document.querySelectorAll('#pub-list .pub-card'));
		var venueRow = document.getElementById('pub-filter-venues');
		var emptyMsg = document.getElementById('pub-filter-empty');
		var type = '';
		var venue = '';

		function setActive(group, attr, value) {
			root.querySelectorAll('[data-filter-group="' + group + '"] .pub-filter-btn').forEach(function (btn) {
				var v = btn.getAttribute(attr) || '';
				btn.classList.toggle('is-active', v === value);
			});
		}

		function apply() {
			var visible = 0;
			cards.forEach(function (card) {
				var cardType = card.getAttribute('data-type') || '';
				var cardVenue = card.getAttribute('data-venue') || '';
				var show = true;
				if (type === 'conference' && venue === 'nature') {
					show = cardVenue === 'nature';
				} else if (type && cardType !== type) {
					show = false;
				} else if (type === 'conference' && venue && cardVenue !== venue) {
					show = false;
				}
				card.classList.toggle('is-filtered-out', !show);
				card.classList.remove('is-even-visible');
				if (show) {
					visible += 1;
					if (visible % 2 === 0) card.classList.add('is-even-visible');
					card.classList.add('in');
				}
			});
			if (emptyMsg) emptyMsg.hidden = visible > 0;
		}

		root.addEventListener('click', function (e) {
			var btn = e.target.closest('.pub-filter-btn');
			if (!btn || !root.contains(btn)) return;

			if (btn.hasAttribute('data-type')) {
				type = btn.getAttribute('data-type') || '';
				venue = '';
				setActive('type', 'data-type', type);
				setActive('venue', 'data-venue', '');
				if (venueRow) venueRow.hidden = type !== 'conference';
			} else if (btn.hasAttribute('data-venue')) {
				venue = btn.getAttribute('data-venue') || '';
				setActive('venue', 'data-venue', venue);
			}
			apply();
		});

		apply();
	})();
})();
