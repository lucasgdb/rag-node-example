import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { readFile } from "node:fs/promises";
import { ChatOllama, OllamaEmbeddings } from "@langchain/ollama";

const MODEL = process.env.NLP_MODEL;
const BASE_URL = process.env.OLLAMA_BASE_URL;

const model = new ChatOllama({
    model: MODEL,
    baseURL: BASE_URL,
    temperature: 0,
    maxRetries: 2,
    streaming: true,
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
        throw new Error("Insufficient context to answer.");
    }

    const context = hits.map((d) => d.pageContent).join("\n\n");
    const prompt = `
        You are an assistant specialized in JavaScript. Rely strictly on context—no outside info.

        Context:
        ${context}

        Question:
        ${question}

        Instructions:
        - Answer in max 3 sentences.
    `;

    return model.stream(prompt);
}

const questions = [
    "Is JavaScript an object-oriented programming language?",
    "Is JavaScript an interpreted language?",
    "Are Node.js and JavaScript the same?",
];

for (const question of questions) {
    console.log(`\n❓ ${question}`);
    process.stdout.write("💡 Answer: ");

    try {
        const stream = await answerQuestion(question);

        for await (const chunk of stream) {
            process.stdout.write(chunk.content);
        }

        console.log(); // newline after answer
    } catch (err) {
        console.log("\n⚠️", err.message);
    }
}
