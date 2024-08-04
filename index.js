const express = require('express');
const axios = require('axios');
const cheerio = require('cheerio');
const tf = require('@tensorflow/tfjs-node');
const { pipeline } = require('@huggingface/transformers');

const app = express();
const port = process.env.PORT || 3000;

async function fetchLinks(url) {
    const response = await axios.get(url);
    const html = response.data;
    const $ = cheerio.load(html);
    const links = [];

    $('a').each((index, element) => {
        const href = $(element).attr('href');
        if (href) {
            links.push(href);
        }
    });

    return links;
}

const loadModel = async () => {
    return await pipeline('feature-extraction', 'xlm-roberta-large');
};

const encodeText = async (model, text) => {
    const embeddings = await model(text);
    return embeddings[0][0];
};

const cosineSimilarity = (vecA, vecB) => {
    const dotProduct = tf.dot(vecA, vecB).dataSync()[0];
    const magnitudeA = tf.norm(vecA).dataSync()[0];
    const magnitudeB = tf.norm(vecB).dataSync()[0];
    return dotProduct / (magnitudeA * magnitudeB);
};

const evaluateLinks = async (links, searchQuery) => {
    const model = await loadModel();
    const searchQueryEmbedding = await encodeText(model, searchQuery);

    const linkEvaluations = await Promise.all(links.map(async (link) => {
        try {
            const response = await axios.get(link);
            const html = response.data;
            const $ = cheerio.load(html);
            const pageText = $('body').text();
            const pageEmbedding = await encodeText(model, pageText);
            const similarity = cosineSimilarity(searchQueryEmbedding, pageEmbedding);
            return { link, similarity };
        } catch (error) {
            return { link, similarity: -1 };
        }
    }));

    return linkEvaluations.sort((a, b) => b.similarity - a.similarity);
};

app.get('/crawl', async (req, res) => {
    const { query, urls } = req.query;
    const urlArray = urls.split(',');

    const allLinks = [];

    for (const url of urlArray) {
        const links = await fetchLinks(url);
        allLinks.push(...links);
    }

    const uniqueLinks = [...new Set(allLinks)];
    const rankedLinks = await evaluateLinks(uniqueLinks, query);

    res.json(rankedLinks);
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
