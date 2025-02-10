import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatDeepSeek } from "@langchain/deepseek";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolMessage } from "@langchain/core/messages";
import { config } from "dotenv";
// sk-f392fbac415e4ebab54b193536f94177

config();

const llm = new ChatDeepSeek({
    apiKey: process.env.DEEPSEEK_API_KEY,
    model: "deepseek-reasoner",
});

const multiply = tool(
    async ({ a, b }) => {
        return a * b;
    },
    {
        name: "multiply",
        description: "Multiply two numbers",
        schema: z.object({
            a: z.number().describe("first number"),
            b: z.number().describe("second number"),
        }),
    }
);

const add = tool(
    async ({ a, b }) => {
        return a + b;
    },
    {
        name: "add",
        description: "Add two numbers together",
        schema: z.object({
            a: z.number().describe("first number"),
            b: z.number().describe("second number"),
        }),
    }
);

const subtract = tool(
    async ({ a, b }) => {
        return a - b;
    },
    {
        name: "subtract",
        description: "Subtract two numbers",
        schema: z.object({
            a: z.number().describe("first number"),
            b: z.number().describe("second number"),
        }),
    }
);

const divide = tool(
    async ({ a, b }) => {
        return a / b;
    },
    {
        name: "divide",
        description: "Divide two numbers",
        schema: z.object({
            a: z.number().describe("first number"),
            b: z.number().describe("second number"),
        }),
    }
);

const tools = [add, subtract, multiply, divide];

const toolsByName = tools.reduce((acc, tool) => {
    acc[tool.name] = tool;
    return acc;
}, {});

const llmWithTools = llm.bindTools(tools);

async function llmCall(state) {
    const result = await llmWithTools.invoke([
        {
            role: "system",
            content:
                "You are a helpful assistant tasked with performing aritmethic on a pair of numbers.",
        },
        ...state.messages,
    ]);

    return {
        messages: [result],
    };
}

async function toolNode(state) {
    const result = [];
    const lastMessage = state.messages.at(-1);
    if (lastMessage?.tool_calls?.lenght) {
        for (const toolCall of lastMessage.tool_calls) {
            const tool = toolsByName[toolCall.tool];
            const observation = await tool.invoke(toolCall.args);
            result.push(
                new ToolMessage({
                    content: observation,
                    tool_call_id: toolCall.id,
                })
            );
        }
    }
    return {
        messages: result,
    };
}

function ShouldContinue(state) {
    const messages = state.messages;
    const lastMessage = messages.at(-1);

    if (lastMessage?.tool_calls?.length) {
        return "Action";
    }

    return "__end__";
}

const agentBuilder = new StateGraph(MessagesAnnotation)
    .addNode("llm", llmCall)
    .addNode("tool", toolNode)
    .addEdge("__start__", "llm")
    .addConditionalEdges("llm", ShouldContinue, {
        Action: "tool",
        __end__: "__end__",
    })
    .addEdge("tool", "llm")
    .compile();

const messages = [{ role: "user", content: "What is 2 + 2?" }];

const result = await agentBuilder.invoke({ messages });

console.log(result.messages);
