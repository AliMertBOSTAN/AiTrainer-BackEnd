const path = require('path');
const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const { connectDatabase } = require('./config/db');
const authRoutes = require('./routes/authRoutes');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const app = express();
const port = process.env.PORT || 5000;
const clientOrigin = process.env.CLIENT_ORIGIN;

app.use(express.json({ limit: '1mb' }));
app.use(
  cors(
    clientOrigin
      ? {
          origin: clientOrigin.split(',').map((origin) => origin.trim()),
          credentials: true,
        }
      : undefined
  )
);

app.get('/health', (req, res) => {
  res.json({ status: 'ok', uptime: process.uptime() });
});

app.use('/api/auth', authRoutes);

app.use((req, res) => {
  res.status(404).json({ message: 'Endpoint bulunamadı.' });
});

async function start() {
  try {
    await connectDatabase(process.env.MONGODB_URI);
    app.listen(port, () => {
      console.log(`🚀 Auth server running on http://localhost:${port}`);
    });
  } catch (error) {
    console.error('Server failed to start:', error.message);
    process.exit(1);
  }
}

start();
