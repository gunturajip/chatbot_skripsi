const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const inputInitHeight = chatInput.scrollHeight;

let isNewResponse = false;
let userMessage = null; // Variable to store user's message
let botMessage = null;
let conversationCount = 0;

const socket = io.connect('http://localhost:5000');  // Replace with your server address
socket.on('connect', () => {
    console.log('Connected to the WebSocket server');
});

const createChatLi = (message, className, currentMillis) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").textContent = message;
    chatLi.querySelector("p").classList.add(currentMillis);
    return chatLi; // return chat <li> element
}

const handleChat = async () => {
    userMessage = chatInput.value.trim(); // Get user entered message and remove extra whitespace
    if (!userMessage) return;

    // Clear the input textarea and set its height to default
    chatInput.value = "";
    chatInput.style.height = `${inputInitHeight}px`;

    // Append the user's message to the chatbox
    const outgoingCurrentMillis = "_" + Date.now();
    chatbox.appendChild(createChatLi(userMessage, "outgoing", outgoingCurrentMillis));
    chatbox.scrollTo(0, chatbox.scrollHeight);

    const incomingCurrentMillis = "_" + Date.now();
    const incomingChatLi = createChatLi("Berpikir...", "incoming", incomingCurrentMillis);
    chatbox.appendChild(incomingChatLi);
    chatbox.scrollTo(0, chatbox.scrollHeight);

    try {
        socket.emit('process_text', JSON.stringify({
            message: userMessage,
        }));

        // Wait for the response with a timeout of 100 milliseconds (adjust as needed)
        const response = await waitForWebSocketResponse(500);

        if (response !== null) {
            incomingChatLi.querySelector("p." + incomingCurrentMillis).textContent = response;
            conversationCount++;
        } else {
            // Handle the case when the response is null (timeout occurred)
            incomingChatLi.querySelector("p." + incomingCurrentMillis).textContent = "Response timeout";
            conversationCount++;
        }
    } catch (error) {
        incomingChatLi.querySelector("p." + incomingCurrentMillis).textContent = "Ups! Ada kendala teknis. Mohon mencoba kembali.";
        conversationCount++;
    }
}

// Function to wait for WebSocket response with a timeout
const waitForWebSocketResponse = (timeout) => {
    return new Promise((resolve, reject) => {
        const timeoutId = setTimeout(() => {
            // Reject the promise if the timeout occurs
            clearTimeout(timeoutId);
            reject('WebSocket response timeout');
        }, timeout);

        // Event listener for receiving responses from the WebSocket server
        socket.on('response_text', (response) => {
            // Resolve the promise when a response is received
            clearTimeout(timeoutId);
            resolve(response);
        });
    });
};

chatInput.addEventListener("input", () => {
    // Adjust the height of the input textarea based on its content
    chatInput.style.height = `${inputInitHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    // If Enter key is pressed without Shift key and the window 
    // width is greater than 800px, handle the chat
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleChat();
    }
});

sendChatBtn.addEventListener("click", handleChat);
closeBtn.addEventListener("click", () => {
    document.body.classList.remove("show-chatbot");
    console.log(userMessage);
});
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));