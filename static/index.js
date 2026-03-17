(function() {
  const STORAGE_KEY = 'selectedCustomerIds';

  function loadSelected() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) {
      return [];
    }
  }

  function saveSelected(ids) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(ids));
    } catch (e) {}
  }

  function updateCounter(ids) {
    const el = document.getElementById('selected-counter');
    if (!el) return;
    if (!ids.length) {
      el.textContent = '（已选 0 位客户）';
    } else {
      el.textContent = '（已选 ' + ids.length + ' 位客户，将在对比页展示）';
    }
  }

  document.addEventListener('DOMContentLoaded', function() {
    const checkboxes = Array.from(document.querySelectorAll('input[name="ids"][type="checkbox"]'));
    const form = document.querySelector('.compare-form');
    let selected = loadSelected();

    // 初始化勾选状态
    checkboxes.forEach(cb => {
      if (selected.includes(cb.value)) cb.checked = true;
    });
    updateCounter(selected);

    // 勾选/取消时更新本地存储
    checkboxes.forEach(cb => {
      cb.addEventListener('change', function() {
        const id = this.value;
        if (this.checked) {
          if (!selected.includes(id)) selected.push(id);
        } else {
          selected = selected.filter(x => x !== id);
        }
        saveSelected(selected);
        updateCounter(selected);
      });
    });

    // 提交到 /compare 时，把所有已选 ID 作为隐藏字段带上（包括不在当前页的）
    if (form) {
      form.addEventListener('submit', function() {
        Array.from(form.querySelectorAll('input[type="hidden"][name="ids"]')).forEach(el => el.remove());
        selected.forEach(id => {
          const hidden = document.createElement('input');
          hidden.type = 'hidden';
          hidden.name = 'ids';
          hidden.value = id;
          form.appendChild(hidden);
        });
      });
    }
  });
})();

