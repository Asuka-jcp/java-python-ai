const platformsEl = document.getElementById('platforms');
const topicListEl = document.getElementById('topicList');
const topicNoteEl = document.getElementById('topicNote');
const fetchBtn = document.getElementById('fetchBtn');

const inputTextEl = document.getElementById('inputText');
const outputTextEl = document.getElementById('outputText');
const rewriteNoteEl = document.getElementById('rewriteNote');
const rewriteBtn = document.getElementById('rewriteBtn');

const chatBoxEl = document.getElementById('chatBox');
const chatInputEl = document.getElementById('chatInput');
const chatSendBtn = document.getElementById('chatSendBtn');

let selectedPlatform = null;

function appendChat(role, text, note = '') {
    const wrapper = document.createElement('div');
    wrapper.className = `chat-msg ${role}`;

    const content = document.createElement('div');
    content.className = 'chat-content';
    content.textContent = text;

    wrapper.appendChild(content);

    if (note) {
        const noteEl = document.createElement('div');
        noteEl.className = 'chat-note';
        noteEl.textContent = note;
        wrapper.appendChild(noteEl);
    }

    chatBoxEl.appendChild(wrapper);
    chatBoxEl.scrollTop = chatBoxEl.scrollHeight;
}

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

chatSendBtn.onclick = async () => {
    const message = chatInputEl.value.trim();
    if (!message) {
        return;
    }

    appendChat('user', message);
    chatInputEl.value = '';

    const loadingMark = document.createElement('div');
    loadingMark.className = 'chat-msg assistant';
    loadingMark.textContent = '思考中...';
    chatBoxEl.appendChild(loadingMark);

    try {
        const resp = await fetch('/api/gateway/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({message})
        });

        chatBoxEl.removeChild(loadingMark);

        if (!resp.ok) {
            appendChat('assistant', '请求失败，请稍后重试。');
            return;
        }

        const data = await resp.json();
        appendChat('assistant', data.answer || '我暂时没有回答。', `意图: ${data.intent || 'unknown'} ${data.note || ''}`);
    } catch (e) {
        chatBoxEl.removeChild(loadingMark);
        appendChat('assistant', '网络异常，请检查服务是否启动。');
    }
};

chatInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatSendBtn.click();
    }
});

appendChat('assistant', '你好，我可以帮你查询热点，也可以改写文章。');
loadPlatforms();
