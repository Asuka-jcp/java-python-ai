const platformsEl = document.getElementById('platforms');
const topicListEl = document.getElementById('topicList');
const topicNoteEl = document.getElementById('topicNote');
const fetchBtn = document.getElementById('fetchBtn');

const inputTextEl = document.getElementById('inputText');
const outputTextEl = document.getElementById('outputText');
const rewriteNoteEl = document.getElementById('rewriteNote');
const rewriteBtn = document.getElementById('rewriteBtn');

let selectedPlatform = null;

async function loadPlatforms() {
    const resp = await fetch('/api/gateway/platforms');
    const platforms = await resp.json();
    platformsEl.innerHTML = '';
    platforms.forEach(p => {
        const btn = document.createElement('button');
        btn.className = 'chip';
        btn.textContent = p;
        btn.onclick = () => {
            selectedPlatform = p;
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
        };
        platformsEl.appendChild(btn);
    });
}

fetchBtn.onclick = async () => {
    if (!selectedPlatform) {
        alert('请先选择平台');
        return;
    }
    topicNoteEl.textContent = '加载中...';
    topicListEl.innerHTML = '';
    const resp = await fetch(`/api/gateway/hot-topics?platform=${encodeURIComponent(selectedPlatform)}`);
    if (!resp.ok) {
        topicNoteEl.textContent = '获取失败';
        return;
    }
    const data = await resp.json();
    topicNoteEl.textContent = `${data.platform} ${data.date} 热点（${data.note || ''}）`;
    (data.topics || []).forEach(t => {
        const li = document.createElement('li');
        li.innerHTML = `<a target="_blank" href="${t.url}">${t.title}</a>`;
        topicListEl.appendChild(li);
    });
};

rewriteBtn.onclick = async () => {
    const text = inputTextEl.value.trim();
    if (text.length < 50) {
        alert('文章至少50字');
        return;
    }
    rewriteNoteEl.textContent = '改写中...';
    outputTextEl.value = '';
    const resp = await fetch('/api/gateway/rewrite', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
    });
    if (!resp.ok) {
        rewriteNoteEl.textContent = '改写失败';
        return;
    }
    const data = await resp.json();
    rewriteNoteEl.textContent = data.note || '';
    outputTextEl.value = data.rewrittenText || '';
};

loadPlatforms();
