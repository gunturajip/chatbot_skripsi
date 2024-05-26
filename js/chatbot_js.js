const chatbotToggler = document.querySelector(".chatbot-toggler");
const closeBtn = document.querySelector(".close-btn");
const chatbox = document.querySelector(".chatbox");
const chatInput = document.querySelector(".chat-input textarea");
const sendChatBtn = document.querySelector(".chat-input span");
const inputInitHeight = chatInput.scrollHeight;

let userMessage = null; // Variable to store user's message
let conversationCount = 0;
let answers;
let stopwords;
let words;
let classes;
let model;

window.onload = async () => {
    answers = await fetch(`${window.location.origin}/answers.json`)
        .then((response) => response.json());
    stopwords = await fetch(`${window.location.origin}/stopwords.txt`)
        .then((response) => response.text())
        .then((text) => text.trim().split('\n'));
    words = await fetch(`${window.location.origin}/words.txt`)
        .then((response) => response.text())
        .then((text) => text.trim().split('\r\n'));
    classes = await fetch(`${window.location.origin}/classes.txt`)
        .then((response) => response.text())
        .then((text) => text.trim().split('\r\n'));
    model = await tf.loadLayersModel(`${window.location.origin}/chatbot/relu_sgd/model.json`);
}

const createChatLi = (message, className, currentMillis) => {
    // Create a chat <li> element with passed message and className
    const chatLi = document.createElement("li");
    chatLi.classList.add("chat", `${className}`);
    let chatContent = className === "outgoing" ? `<p></p>` : `<span class="material-symbols-outlined">smart_toy</span><p></p>`;
    chatLi.innerHTML = chatContent;
    chatLi.querySelector("p").innerHTML = message;
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
        async function stopword_kalimat(str, stopwords) {
            let result = "";
            await str.split(" ").forEach(word => {
                if (!stopwords.includes(word)) {
                    result += word + " ";
                }
            });
            return result.trim();
        }

        async function clean_up_sentence(sentence, stopwords) {
            sentence = await sentence.replace(/[^\w\s]|_/g, " "); // SPECIAL CHARACTERS REMOVAL
            sentence = await sentence.toLowerCase().trim(); // CASE FOLDING
            sentence = await stopword_kalimat(sentence, stopwords); // STOPWORDS REMOVAL
            sentence = await sentence.split(" "); // TOKENIZING
            return sentence
        }

        async function bag_of_words(sentence, stopwords, words) {
            const sentence_words = await clean_up_sentence(sentence, stopwords);
            let bag = Array(461).fill(0);
            for (let w of sentence_words) {
                for (let i = 0; i < 461; i++) {
                    if (words[i] === w) {
                        bag[i] = 1;
                    }
                }
            }
            return bag;
        }

        async function prepare() {
            let input = await bag_of_words(userMessage, stopwords, words);
            return input
        }

        async function predict_class() {
            const input = await prepare();
            const prediction = await model.predict(tf.tensor(input, [1, 461])).data();
            const probThreshold = 0.25;
            let result = [];
            for (let i = 0; i < prediction.length; i++) {
                if (prediction[i] > probThreshold) result.push([i, prediction[i]]);
            }
            result.sort((a, b) => b[1] - a[1]);
            return result;
        }

        async function get_response(intents_list) {
            let tag = classes[intents_list[0][0]];
            if ((tag === "salam_pembuka" || tag === "salam_penutup") && intents_list.length > 1) {
                tag = classes[intents_list[1][0]];
            }
            let result = "";
            for (let i of answers["answers"]) {
                if (i["tag"] === tag) {
                    result = i["responses"][0];
                    break;
                }
            }
            return result;
        }

        // Wait for the response with a timeout of 100 milliseconds (adjust as needed)
        const prediction = await predict_class();
        const response = await get_response(prediction);

        if (response !== null) {
            incomingChatLi.querySelector("p." + incomingCurrentMillis).innerHTML = response;
            conversationCount++;
        } else {
            // Handle the case when the response is null (timeout occurred)
            incomingChatLi.querySelector("p." + incomingCurrentMillis).innerHTML = "Response timeout";
            conversationCount++;
        }
    } catch (error) {
        incomingChatLi.querySelector("p." + incomingCurrentMillis).innerHTML = "Ups! Ada kendala teknis. Mohon mencoba kembali.";
        conversationCount++;
    }
}

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
});
chatbotToggler.addEventListener("click", () => document.body.classList.toggle("show-chatbot"));