/**
 * Sidebar name, job title, and avatar from /api/me (LinkedIn-backed RecruiterUser).
 * Auto-loads on pages with #sidebarName but without the Settings form (#profileFullName).
 */
(function () {
    function initials(name) {
        if (!name || !String(name).trim()) return '?';
        var p = String(name).trim().split(/\s+/);
        if (p.length === 1) return p[0].slice(0, 2).toUpperCase();
        return (p[0][0] + p[p.length - 1][0]).toUpperCase();
    }

    function applySidebarFromUser(me) {
        var nameEl = document.getElementById('sidebarName');
        var titleEl = document.getElementById('sidebarTitle');
        if (!nameEl || !titleEl) return;
        nameEl.textContent = me.full_name || me.email || 'Recruiter';
        titleEl.textContent = me.job_title || me.email || '—';

        var wrap = document.getElementById('sidebarAvatar');
        var avText = document.getElementById('sidebarAvatarText');
        if (!wrap || !avText) return;

        if (me.picture_url) {
            avText.style.display = 'none';
            var img = document.getElementById('sidebarAvatarImg');
            if (!img) {
                img = document.createElement('img');
                img.id = 'sidebarAvatarImg';
                img.className = 'w-full h-full object-cover rounded-full';
                img.alt = '';
                wrap.appendChild(img);
            }
            img.src = me.picture_url;
            img.style.display = 'block';
        } else {
            var existing = document.getElementById('sidebarAvatarImg');
            if (existing) existing.style.display = 'none';
            avText.style.display = 'inline';
            avText.textContent = initials(me.full_name || me.email);
        }
    }

    async function loadSidebarProfile() {
        var res = await fetch('/api/me', { credentials: 'same-origin' });
        if (res.status === 401) {
            window.location.href = '/login';
            return;
        }
        if (!res.ok) return;
        var me = await res.json();
        applySidebarFromUser(me);
    }

    window.HireFastApplySidebarFromUser = applySidebarFromUser;
    window.HireFastLoadSidebarProfile = loadSidebarProfile;

    document.addEventListener('DOMContentLoaded', function () {
        if (document.getElementById('sidebarName') && !document.getElementById('profileFullName')) {
            loadSidebarProfile();
        }
    });
})();
