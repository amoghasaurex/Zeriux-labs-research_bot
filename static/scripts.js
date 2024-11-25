document.addEventListener("DOMContentLoaded", () => {
    // Link the login and register links
    document.getElementById("login-link").addEventListener("click", showLogin);
    document.getElementById("register-link").addEventListener("click", showRegister);

    // Clear error messages on form focus
    document.querySelectorAll("input").forEach(input => {
        input.addEventListener("focus", () => {
            document.getElementById("login-error").innerText = "";
            document.getElementById("register-error").innerText = "";
        });
    });
});

function showLogin() {
    document.getElementById("register-section").classList.add("hidden");
    document.getElementById("login-section").classList.remove("hidden");
}

function showRegister() {
    document.getElementById("login-section").classList.add("hidden");
    document.getElementById("register-section").classList.remove("hidden");
}

function showHome(username) {
    document.getElementById("username-display").innerText = username;
    document.getElementById("home-section").classList.remove("hidden");
    document.getElementById("login-section").classList.add("hidden");
    document.getElementById("register-section").classList.add("hidden");
}

async function login(event) {
    event.preventDefault();
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;

    const response = await fetch('/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });

    const result = await response.json();
    if (result.success) {
        showHome(username);
    } else {
        document.getElementById('login-error').innerText = result.error;
    }
}

async function register(event) {
    event.preventDefault();
    const username = document.getElementById('register-username').value;
    const password = document.getElementById('register-password').value;

    const response = await fetch('/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
    });

    const result = await response.json();
    if (result.success) {
        showHome(username);
    } else {
        document.getElementById('register-error').innerText = result.error;
    }
}

async function logout() {
    await fetch('/logout', { method: 'POST' });
    showLogin();
}

async function uploadFile(event) {
    event.preventDefault();
    const fileInput = document.getElementById('file-input');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/upload', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    if (result.error) {
        document.getElementById('upload-error').innerText = result.error;
    } else {
        alert("File uploaded and text extracted!");
    }
}

async function sendMessage() {
    const userInput = document.getElementById('chat-input').value;
    const response = await fetch('/get_response', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ msg: userInput })
    });

    const result = await response.text();
    const chatOutput = document.getElementById('chat-output');
    chatOutput.innerHTML += `<p><b>You:</b> ${userInput}</p><p><b>Bot:</b> ${result}</p>`;
    document.getElementById('chat-input').value = "";
    chatOutput.scrollTop = chatOutput.scrollHeight;
}
