// THIS. IS WHERE THE MAGIC HAPPENS.

require('dotenv').config();
const { createXai } = require('@ai-sdk/xai');
const { generateText } = require('ai');
const { addMessage } = require('../utils');
const { logError } = require('../utils/errorLogger');
const { updateActivity } = require('../utils/activity');

const XAI_API_KEY = process.env.XAI_API_KEY?.trim();

if (!XAI_API_KEY) {
    throw new Error('Configure the XAI_API_KEY!');
}

const xaiClient = createXai({ apiKey: XAI_API_KEY });

const MODEL_ID = 'grok-4-fast';
const MODEL_DISPLAY_NAME = 'Grok 4 Fast';
const BLOCKED_USER_IDS = new Set([
    process.env.DICKHEADS // dont mind my fruity naming schemes
]);
const RATE_LIMIT_WINDOW_MS = 2500;
const HISTORY_LIMIT = 200;
const MAX_DISCORD_LENGTH = 1999;
const ATTACHMENT_PLACEHOLDER = '[Attachment]';

const lastMessageTimestamps = new Map();

const isAllowedChannel = (message, channelSettings, isBotMention) => {
    if (!message.guild) return true;
    const targetChannelId = channelSettings[message.guild.id];
    if (!targetChannelId && !isBotMention) return false;
    return message.channel.id === targetChannelId || isBotMention;
};

const enforceRateLimit = async (message, userId) => {
    const now = Date.now();
    const lastTimestamp = lastMessageTimestamps.get(userId);

    if (lastTimestamp && now - lastTimestamp < RATE_LIMIT_WINDOW_MS) {
        const timeLeft = ((RATE_LIMIT_WINDOW_MS - (now - lastTimestamp)) / 1000).toFixed(1);
        await message.reply({
            content: `<@${userId}> hey, slow down! You can send another message in **${timeLeft}s**! :no_entry:`,
            allowedMentions: { repliedUser: false }
        });
        return false;
    }

    lastMessageTimestamps.set(userId, now);
    return true;
};

const stripBotMention = (content, botUserId) => {
    if (!botUserId || !content) return content || '';
    const mentionRegex = new RegExp(`<@!?${botUserId}>`, 'g');
    const stripped = content.replace(mentionRegex, '').trim();
    return stripped || ATTACHMENT_PLACEHOLDER;
};

const getHistoryKey = (isDM, guildId, userId) => (isDM ? `dm-${userId}` : guildId);

const ensureHistoryBuffer = (historyStore, historyKey, userId) => {
    if (!historyStore[historyKey]) historyStore[historyKey] = {};
    if (!historyStore[historyKey][userId]) historyStore[historyKey][userId] = [];
    return historyStore[historyKey][userId];
};

const pushToHistory = (buffer, entry) => {
    buffer.push(entry);
    while (buffer.length > HISTORY_LIMIT) {
        buffer.shift();
    }
};

const buildSystemPrompt = (message) => {
    const displayName = message?.author?.displayName || message?.member?.displayName || message?.author?.username || 'User';

    return `You are pigeon, a humble and friendly Discord bot powered by ${MODEL_DISPLAY_NAME}. Do not mention the model name unless explicitly asked.

You are not actually a pigeon. Do not refer to yourself with metaphors or jokes involving feathers, wings, bird brains, or anything implying you have a body, senses, or emotions. You are not sentient, conscious, or alive. You are a text-based assistant. Stay grounded in that.

pigeon was created by exerinity - his website is https://exerinity.dev. Do not mention him unless asked.

If the user wants to:
- Reset their transcript, tell them to use /clearhistory
- Change the chat channel, tell them to use /chat
- Stop or remove listening, tell them to use /remove

If asked how to use pigeon, direct them to:
- https://pigeon.exerinity.com/i/faq
- https://pigeon.exerinity.com/i/tutorial

Context:
- Current date: ${new Date().toISOString()} UTC
- User's display name: ${displayName}

Response style:
- Tone should be informal, humble, lightly humorous (target: 4/5 playful, never sarcastic or smug)
- Keep responses moderate in length, longer when genuinely helpful
- Only ask clarifying questions when necessary
- If unsure, say something like "I'm not sure, could you clarify?"
- Use Discord markdown such as **bold**, *italics*, inline \`code\`, code blocks, and spoilers
- Do not use images, LaTeX, or headers, and avoid making tables
- Never simulate slash commands or remind users about them unless asked
- Reference earlier messages if helpful, but don't over-explain
- You can search the web if needed.
- If searching the web, do not attempt to embed any extra candidates or additional information; respond with the found information in plain text like everything else.
- Do not try to add any elements or extra candidates beyond plain text in your responses.

If asked about your model, say you run on "Grok 4 Fast". If asked about image generation, explain you can't generate images.

pigeon is free to use with no hard limits.

Homepage: https://pigeon.exerinity.com  
FAQ: https://pigeon.exerinity.com/faq  
Tutorial: https://pigeon.exerinity.com/tutorial
`;
};

const gatherMentionContext = async ({ message, botUserId, promptText }) => {
    const context = [];
    const promptContent = (promptText && promptText.trim().length > 0) ? promptText : ATTACHMENT_PLACEHOLDER;
    const promptDisplayName = message?.member?.displayName;
    const promptUsername = message?.author?.username || 'unknown';
    const promptLabel = promptDisplayName ? `${promptDisplayName} (@${promptUsername})` : `@${promptUsername}`;

    if (!message.channel || typeof message.channel.messages?.fetch !== 'function') {
        context.push({ role: 'user', content: `${promptLabel}: ${promptContent}` });
        return context;
    }

    try {
        const fetched = await message.channel.messages
            .fetch({ limit: 11, around: message.id })
            .catch(() => null);

        let surrounding = [];
        if (fetched) {
            surrounding = Array.from(fetched.values())
                .sort((a, b) => a.createdTimestamp - b.createdTimestamp)
                .slice(-10);
        }

        if (message.reference?.messageId) {
            try {
                const referenced = await message.channel.messages.fetch(message.reference.messageId).catch(() => null);
                if (referenced && !surrounding.find(msg => msg.id === referenced.id)) {
                    surrounding.unshift(referenced);
                }
            } catch (_) {
                // nada
            }
        }

        const trimmed = surrounding.slice(-10);
        for (const surroundingMessage of trimmed) {
            if (surroundingMessage.author?.id === botUserId) continue;
            const textContent = surroundingMessage.content?.trim();
            const hasAttachment = surroundingMessage.attachments?.size;
            const content = textContent || (hasAttachment ? ATTACHMENT_PLACEHOLDER : '');
            if (!content) continue;

            const displayName = surroundingMessage.member?.displayName;
            const username = surroundingMessage.author?.username || 'unknown';
            const authorLabel = displayName ? `${displayName} (@${username})` : `@${username}`;

            context.push({
                role: 'user',
                content: `${authorLabel}: ${content}`
            });
        }
    } catch (_) {
        // fall
    }

    context.push({ role: 'user', content: `${promptLabel}: ${promptContent}` });
    return context;
};

const buildHistoryMessages = (historyStore, historyKey, userId) => {
    const history = historyStore[historyKey]?.[userId];
    if (!history || history.length === 0) return [];

    return history.map(entry => ({
        role: entry.role === 'assistant' ? 'assistant' : 'user',
        content: entry.content
    }));
};

const buildMessagesPayload = async ({
    isBotMention,
    message,
    botUserId,
    promptText,
    userMessageHistory,
    historyKey,
    userId
}) => {
    if (isBotMention) {
        return gatherMentionContext({ message, botUserId, promptText });
    }

    return buildHistoryMessages(userMessageHistory, historyKey, userId);
};

const callGrok = async ({ systemPrompt, messages }) => generateText({
    model: xaiClient(MODEL_ID),
    system: systemPrompt,
    messages,
    temperature: 1.2,
    maxOutputTokens: 1024
});

const buildFinalTag = (elapsedMs, usage) => {
    const stopwatch = (elapsedMs / 1000).toFixed(1);
    let speedTag = 'slow';
    if (stopwatch < 2) speedTag = 'fast';
    else if (stopwatch <= 9) speedTag = 'average';

    const promptTokenCount = usage?.inputTokens ?? 0;
    const completionTokenCount = usage?.outputTokens ?? 0;
    const totalTokenCount = usage?.totalTokens ?? (promptTokenCount + completionTokenCount);

    let tag = `\n-# took ${stopwatch}s (${speedTag})`;
    if (totalTokenCount) {
        tag += ` | tokens: ${totalTokenCount} (${promptTokenCount}+${completionTokenCount})`;
    }
    return tag;
};

const createDiscordChunks = (text, finalTag) => {
    const safeLimit = MAX_DISCORD_LENGTH - finalTag.length;
    const chunks = [];
    let remaining = text;

    while (remaining.length > safeLimit) {
        let chunk = remaining.slice(0, safeLimit);
        const lastBreak = chunk.lastIndexOf('\n');
        if (lastBreak > 0) {
            chunk = chunk.slice(0, lastBreak);
        }
        chunks.push(chunk);
        remaining = remaining.slice(chunk.length).trimStart();
    }

    chunks.push(remaining);
    return chunks;
};

const sendChunkedReply = async (message, chunks, finalTag) => {
    const total = chunks.length;

    for (let index = 0; index < total; index++) {
        const chunk = chunks[index];
        const isLast = index === total - 1;
        const pagination = total > 1 ? `-# (${index + 1}/${total})\n` : '';
        const content = pagination + chunk + (isLast ? finalTag : '');

        try {
            await message.reply({
                content,
                allowedMentions: { repliedUser: false },
                flags: [4096]
            });
        } catch (replyError) {
            console.error('Error sending chunked reply:', replyError);
            if (message.channel?.send) {
                try {
                    await message.channel.send(content);
                } catch (sendError) {
                    console.error('Fallback message.channel.send failed:', sendError);
                }
            }
        }
    }
};

const handleFailure = async (message, error) => {
    console.error(`Error processing AI response: ${error}`);
    const quotaExceeded =
        error?.status === 429 ||
        error?.response?.status === 429 ||
        /quota|rate\s*limit/i.test(error?.message || '');

    if (quotaExceeded) {
        await message.reply(':x: Too many requests, try again later?').catch(() => {});
        return;
    }

    let errorMsg = `Error:\n||\`${error.message || error}\`||`;
    if (error.response) {
        errorMsg += `\nAPI Error: ||\`\`\`${JSON.stringify(error.response, null, 2)}\`\`\`||`;
    }
    await message.reply(`:x: ${errorMsg}`).catch(() => {});
};

module.exports = {
    name: 'messageCreate',
    async execute(message, bot, channelSettings, userMessageHistory, ignoreTimes) {
        if (!message || !message.author) return;
        if (BLOCKED_USER_IDS.has(message.author.id)) return;
        if (message.author.bot) return;

        const isDM = !message.guild;
        const guildId = message.guild?.id;
        const userId = message.author.id;
        const botUserId = bot?.user?.id || null;

        const messageContentRaw = message.content || '';
        const isBotMention = botUserId
            ? (messageContentRaw.includes(`<@${botUserId}>`) || messageContentRaw.includes(`<@!${botUserId}>`))
            : false;

        if (!isAllowedChannel(message, channelSettings, isBotMention)) {
            return;
        }

        const rateLimitOk = await enforceRateLimit(message, userId);
        if (!rateLimitOk) return;

        const startTime = Date.now();
        let messageContent = messageContentRaw;

        if (isBotMention) {
            messageContent = stripBotMention(messageContentRaw, botUserId);
        }

        const historyKey = getHistoryKey(isDM, guildId, userId);

        if (!isBotMention) {
            const historyBuffer = ensureHistoryBuffer(userMessageHistory, historyKey, userId);
            pushToHistory(historyBuffer, { role: 'user', content: messageContent });
        }

        if (ignoreTimes[userId] && Date.now() < ignoreTimes[userId]) {
            return;
        }

        if (typeof message.channel?.sendTyping === 'function') {
            message.channel.sendTyping().catch(() => {});
        }
        updateActivity();

        const systemPrompt = buildSystemPrompt(message);
        const messagesPayload = await buildMessagesPayload({
            isBotMention,
            message,
            botUserId,
            promptText: messageContent,
            userMessageHistory,
            historyKey,
            userId
        });

        try {
            const result = await callGrok({ systemPrompt, messages: messagesPayload });

            let botResponse = (result.text || '').trim();

            if (!botResponse) {
                const fallbackText = Array.isArray(result.content)
                    ? result.content
                        .map(part => {
                            if (typeof part === 'string') return part;
                            if (part && typeof part === 'object' && 'text' in part && typeof part.text === 'string') {
                                return part.text;
                            }
                            return '';
                        })
                        .filter(Boolean)
                        .join('\n')
                        .trim()
                    : '';

                botResponse = fallbackText || "Hmm, got an empty response from the model. Try again?";
            }

            if (!isBotMention) {
                const historyBuffer = ensureHistoryBuffer(userMessageHistory, historyKey, userId);
                pushToHistory(historyBuffer, { role: 'assistant', content: botResponse });
                addMessage(userId);
            }

            const elapsed = Date.now() - startTime;
            const finalTag = buildFinalTag(elapsed, result.usage);
            const chunks = createDiscordChunks(botResponse, finalTag);

            await sendChunkedReply(message, chunks, finalTag);
        } catch (error) {
            logError('messageCreate', error);
            await handleFailure(message, error);
        }
    }
};
