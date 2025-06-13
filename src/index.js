import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { readFile } from "node:fs/promises";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

const MODEL = process.env.NLP_MODEL
const BASE_URL = process.env.OLLAMA_BASE_URL

const model = new ChatOllama({
    temperature: 0,
    maxRetries: 2,
    model: MODEL,
    baseURL: BASE_URL,
});

const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseURL: BASE_URL,
});

const raw = await readFile("./docs/javascript.txt", "utf-8");

const documents = raw
    .split(".")
    .map((s) => ({ pageContent: s.trim(), metadata: {} }))
    .filter((d) => d.pageContent.length > 10);

const vectorStore = await MemoryVectorStore.fromDocuments(
    documents,
    embeddings
);

async function answerQuestion(question) {
    const hits = await vectorStore.similaritySearch(question, 2);
    if (hits.length === 0) {
        return "Sorry, I couldn't find enough information to answer.";
    }

    const context = hits.map((d) => d.pageContent).join("\n");

    const prompt = `
        You are an assistant specialized in information from JavaScript language.
        Rely strictly on the provided context and do not introduce any external information.
        Maintain confidentiality, accuracy, and clear language suitable for developer professionals.

        Context:
        ${context}

        Question:
        ${question}

        Instructions:
        - Answer concisely and objectively (max 3 sentences).
        - Use appropriate developer terminology when necessary.
        - If the context does not fully cover the question, reply with “Insufficient information in the context.”
    `;

    const resp = await model.invoke(prompt);
    return resp;
}

await Promise.all([
    "Is JavaScript a object oriented programing language?",
    "Is JavaScript an interpreted language?",
    "Node.js and JavaScript are the same?",
].map(async question => {
    const response = await answerQuestion(question);
    console.log("\n💡 Final Answer:\n", response.content);
}));