const express = require('express');
const cheerio = require('cheerio');
const axios = require('axios');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3002;

// Self-learning data store
let learningData = [];

// Scrape YouTube trending page
async function scrapeYouTubeTrending() {
  try {
    const { data } = await axios.get('https://www.youtube.com/feed/trending');
    const $ = cheerio.load(data);
    const titles = [];

    $('a#video-title').each((i, el) => {
      const title = $(el).text().trim();
      if (title) titles.push(title);
    });

    return titles;
  } catch (err) {
    console.error('Failed to scrape YouTube:', err.message);
    return [];
  }
}

// Basic tokenizer
function tokenize(text) {
  return text.toLowerCase().replace(/[^a-z0-9 ]/g, '').split(' ').filter(Boolean);
}

// Feature extraction from scraped titles
function extractFeatures(texts) {
  return texts.map(txt => {
    const tokens = tokenize(txt);
    return [
      tokens.length,
      tokens.filter(t => t.includes('ai')).length,
      tokens.filter(t => t.includes('music')).length,
      tokens.filter(t => t.includes('how')).length
    ];
  });
}

// Create simple NN
function buildModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [4], units: 8, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
  model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });
  return model;
}

// Train neural network
async function trainAI(features) {
  const xs = tf.tensor2d(features);
  const ys = tf.tensor2d(features.map(() => [1])); // fake positive training for now

  const model = buildModel();
  await model.fit(xs, ys, {
    epochs: 20,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(\`Epoch \${epoch}: Loss = \${logs.loss.toFixed(4)} Acc = \${logs.acc?.toFixed(4)}\`);
      }
    }
  });

  await model.save('file://./trained-model');
  xs.dispose();
  ys.dispose();
}

// Main self-learning loop
async function runSelfLearningLoop() {
  const titles = await scrapeYouTubeTrending();
  const features = extractFeatures(titles);
  learningData = [...learningData, ...features];
  await trainAI(features);
  console.log('✅ Self-learning loop complete');
}

// Routes
app.get('/api/ai/status', (req, res) => {
  res.json({
    status: 'autonomous',
    learnedSamples: learningData.length
  });
});

app.get('/api/ai/learn', async (req, res) => {
  await runSelfLearningLoop();
  res.json({ success: true, samplesLearned: learningData.length });
});

app.listen(PORT, () => {
  console.log(\`🤖 Autonomous AI backend running at http://localhost:\${PORT}\`);
  runSelfLearningLoop(); // kick off first loop
});