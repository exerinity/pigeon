// THIS. IS WHERE THE MAGIC HAPPENS.

require('dotenv').config();
const { GoogleGenAI } = require('@google/genai');
const { addMessage } = require('../utils');
const { logError } = require('../utils/errorLogger');
const { updateActivity } = require('../utils/activity');

const API_KEYS = [
    process.env.GEMINI_API_KEY_1,
    process.env.GEMINI_API_KEY_2,
    process.env.GEMINI_API_KEY_3,
    process.env.GEMINI_API_KEY_4
].filter(key => key && key.trim() !== '');

if (API_KEYS.length === 0) {
    throw new Error('Configure the API keys!');
}

const MODEL_NAME = 'gemini-2.5-flash';
const BLOCKED_USER_IDS = new Set([
    process.env.DICKHEADS // dont mind my fruity naming schemes
]);
const RATE_LIMIT_WINDOW_MS = 2500;
const HISTORY_LIMIT = 200;
const MAX_DISCORD_LENGTH = 1999;
const RETRY_EXTRA_ATTEMPTS = 2;
const ATTACHMENT_PLACEHOLDER = '[Attachment]';
const GOOGLE_SEARCH_TOOL = { googleSearch: {} };

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

    return `You are pigeon, a humble and friendly Discord bot powered by ${MODEL_NAME}. Do not mention the model name unless explicitly asked.

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

If asked about your model, say you run on "Gemini 2.5 Flash". If asked about image generation, explain you can't generate images.

pigeon is free to use with no hard limits.

Homepage: https://pigeon.exerinity.com  
FAQ: https://pigeon.exerinity.com/faq  
Tutorial: https://pigeon.exerinity.com/tutorial
`;
};

const gatherMentionContext = async ({ message, botUserId, promptText }) => {
    const context = [];
    if (!message.channel || typeof message.channel.messages?.fetch !== 'function') {
        context.push({ role: 'user', parts: [{ text: `Prompt: ${promptText}` }] });
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
                parts: [{ text: `${authorLabel}: ${content}` }]
            });
        }
    } catch (_) {
        // fall
    }

    context.push({ role: 'user', parts: [{ text: `Prompt: ${promptText}` }] });
    return context;
};

const buildHistoryMessages = (historyStore, historyKey, userId) => {
    const history = historyStore[historyKey]?.[userId];
    if (!history || history.length === 0) return [];

    return history.map(entry => {
        const isAssistant = entry.role === 'assistant';
        return {
            role: isAssistant ? 'model' : 'user',
            parts: [{ text: `${isAssistant ? 'Assistant' : 'User'}: ${entry.content}` }]
        };
    });
};

const buildMessagesPayload = async ({
    systemPrompt,
    isBotMention,
    message,
    botUserId,
    promptText,
    userMessageHistory,
    historyKey,
    userId
}) => {
    const payload = [{ role: 'model', parts: [{ text: systemPrompt }] }];

    if (isBotMention) {
        const mentionContext = await gatherMentionContext({ message, botUserId, promptText });
        payload.push(...mentionContext);
    } else {
        payload.push(...buildHistoryMessages(userMessageHistory, historyKey, userId));
    }

    return payload;
};

const pickApiKey = (usedKeys) => {
    let pool = API_KEYS.filter(key => !usedKeys.has(key));
    if (pool.length === 0) {
        usedKeys.clear();
        pool = [...API_KEYS];
    }
    const index = Math.floor(Math.random() * pool.length);
    return pool[index];
};

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const executeWithRotation = async (messages, discordMessage) => {
    const maxAttempts = API_KEYS.length + RETRY_EXTRA_ATTEMPTS;
    let statusMsg = null;
    const usedKeys = new Set();

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const apiKey = pickApiKey(usedKeys);
        usedKeys.add(apiKey);

        try {
            const ai = new GoogleGenAI({ apiKey });
            const result = await ai.models.generateContent({
                model: MODEL_NAME,
                contents: messages,
                config: { tools: [GOOGLE_SEARCH_TOOL] },
                generationConfig: {
                    temperature: 1.2,
                    maxOutputTokens: 1024
                }
            });

            if (statusMsg) {
                try {
                    await statusMsg.delete().catch(() => {});
                } catch (_) {
                    // n
                }
            }

            return { result };
        } catch (error) {
            console.error(`Attempt ${attempt} failed with key ending ...${apiKey.slice(-6)}:`, error.message || error);

            const isLastAttempt = attempt === maxAttempts;
            const baseText = `:warning: Something went wrong, retrying... (Attempt ${attempt} of ${maxAttempts})`;
            const finalText = isLastAttempt ? `:no_entry: Exhausted all 6 attempts - giving up...` : baseText;

            try {
                if (!statusMsg) {
                    statusMsg = await discordMessage.reply({
                        content: finalText,
                        allowedMentions: { repliedUser: false }
                    });
                } else {
                    await statusMsg.edit(finalText).catch(() => {});
                }
            } catch (notifyErr) {
                console.warn('Failed to send/update retry status:', notifyErr);
            }

            if (!isLastAttempt) {
                await delay(attempt <= 2 ? 800 : 2000);
            } else {
                throw error;
            }
        }
    }

    throw new Error('All keys exhausted');
};


const extractPrimaryCandidate = (response) => {
    if (!response || !Array.isArray(response.candidates) || response.candidates.length === 0) {
        return { candidate: null, text: null };
    }

    const candidate = response.candidates[0] ?? null;
    if (!candidate || typeof candidate !== 'object') {
        return { candidate: null, text: null };
    }

    const parts = Array.isArray(candidate.content?.parts) ? candidate.content.parts : [];
    const textPart = parts.find(part => typeof part?.text === 'string' && part.text.trim().length > 0) || null;

    return {
        candidate,
        text: textPart ? textPart.text.trim() : null
    };
};

const countGroundingChunks = (candidate) => candidate?.groundingMetadata?.groundingChunks?.length ?? 0;

const cleanGroundingCitations = (text) => text
    .replace(/\[\d+\]\(https?:\/\/[^\s)]+\)/g, '')
    .replace(/\[\^?\d+\]/g, '')
    .replace(/\|\d+\|/g, '')
    .replace(/\s+\[\d+(?:,\s*\d+)*\]$/gm, '')
    .replace(/\n\n\[\d+\]:.*$/gm, '')
    .replace(/\n{2,}/g, '\n\n')
    .trim();

const buildFinalTag = (elapsedMs, usage, searchedSites) => {
    const stopwatch = (elapsedMs / 1000).toFixed(1);
    let speedTag = 'slow';
    if (stopwatch < 2) speedTag = 'fast';
    else if (stopwatch <= 9) speedTag = 'average';

    const promptTokenCount = usage?.promptTokenCount ?? 0;
    const candidatesTokenCount = usage?.candidatesTokenCount ?? 0;
    const totalTokenCount = usage?.totalTokenCount ?? 0;

    let tag = `\n-# took ${stopwatch}s (${speedTag}) | tokens: ${totalTokenCount} (${promptTokenCount}+${candidatesTokenCount})`;
    if (searchedSites > 0) {
        tag += ` | searched ${searchedSites} site${searchedSites > 1 ? 's' : ''}`;
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
        error?.response?.error?.code === 429 ||
        error?.response?.error?.status === 'RESOURCE_EXHAUSTED' ||
        /quota/i.test(error?.message || '');

    if (quotaExceeded) {
        await message.reply(':x: Too many requests, try again later?').catch(() => {});
        return;
    }

    let errorMsg = `Error:\n||\`${error.message || error}\`||`;
    if (error.response?.error) {
        errorMsg += `\nAPI Error: ||\`\`\`${JSON.stringify(error.response.error, null, 2)}\`\`\`||`;
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
            systemPrompt,
            isBotMention,
            message,
            botUserId,
            promptText: messageContent,
            userMessageHistory,
            historyKey,
            userId
        });

        try {
            const { result: response } = await executeWithRotation(messagesPayload, message);
            const { candidate: primaryCandidate, text: parsedText } = extractPrimaryCandidate(response);

            let botResponse;
            if (parsedText) {
                botResponse = parsedText;
            } else if (primaryCandidate) {
                botResponse = "Unexpected response structure or empty candidates. Maybe it's rate-limited. Try later?";
            } else {
                console.error('Malformed response: ', JSON.stringify(response, null, 2));
                botResponse = "Hmm, got an empty response from the model. Try again?";
            }

            botResponse = botResponse.trim();
            const searchedSites = countGroundingChunks(primaryCandidate);

            if (searchedSites > 0) {
                botResponse = cleanGroundingCitations(botResponse);
            }

            if (!isBotMention) {
                const historyBuffer = ensureHistoryBuffer(userMessageHistory, historyKey, userId);
                pushToHistory(historyBuffer, { role: 'assistant', content: botResponse });
                addMessage(userId);
            }

            const elapsed = Date.now() - startTime;
            const usage = response?.usageMetadata;
            const finalTag = buildFinalTag(elapsed, usage, searchedSites);
            const chunks = createDiscordChunks(botResponse, finalTag);

            await sendChunkedReply(message, chunks, finalTag);
        } catch (error) {
            logError('messageCreate', error);
            await handleFailure(message, error);
        }
    }
};
