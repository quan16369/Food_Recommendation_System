const fs = require("fs");
const pdf = require("pdf-parse");
const { HfInference } = require("@huggingface/inference");
const readline = require("readline");
const { ChromaClient } = require("chromadb");
const foodItems = require("./FoodDataSet.js");

const hf = new HfInference("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");
const chroma = new ChromaClient();
const collectionName = "recipe_food";

const extractTextFromPDF = async (filePath) => {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdf(dataBuffer);
    const text = data.text.replace(/\n/g, " ").replace(/ +/g, " ");
    return text;
  } catch (err) {
    console.error("Error extracting text from PDF:", err);
    throw err;
  }
};

const generateEmbeddings = async (text) => {
  try {
    const result = await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: text,
    });
    // console.log("Embedding API result:", result); // Log the entire result
    return result;
  } catch (err) {
    console.error("Error converting text to embeddings:", err);
    throw err;
  }
};

const promptUserInput = (query) => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  return new Promise((resolve) =>
    rl.question(query, (answer) => {
      rl.close();
      resolve(answer);
    })
  );
};

const extractIngredients = (text) => {
  const ingredientsPattern = /Ingredients([\s\S]+?)For /i;
  const ingredientsMatch = ingredientsPattern.exec(text);
  if (ingredientsMatch) {
    return ingredientsMatch[1]
      .split(/[^a-zA-Z0-9]+/)
      .map((item) => item.trim().toLowerCase())
      .filter((item) => item.length > 0);
  }
  return [];
};

const storeEmbeddingsInChromaDB = async (foodItems) => {
  const foodEmbeddings = [];
  const metadatas = foodItems.map(() => ({})); // Empty metadata objects

  for (const item of foodItems) {
    const embedding = await generateEmbeddings(
      item.food_ingredients.join(" ").toLowerCase()
    );
    foodEmbeddings.push(embedding);
    // console.log(item.food_ingredients);
  }
  const ids = foodItems.map((_, index) => index.toString());
  const foodTexts = foodItems.map((item) => item.food_name);
  try {
    const collection = await chroma.getOrCreateCollection({
      name: collectionName,
    });

    await collection.add({
      ids: ids,
      documents: foodTexts,
      embeddings: foodEmbeddings,
      metadatas: metadatas,
    });
    console.log("Stored embeddings in Chroma DB.");
  } catch (error) {
    console.error("Error storing embeddings in Chroma DB:", error);
    throw error;
  }
};

const main = async () => {
  try {
    await storeEmbeddingsInChromaDB(foodItems);

    // Extract and process the recipe PDF
    const filePath = await promptUserInput(
      "Enter the path to the recipe PDF: "
    );
    const text = await extractTextFromPDF(filePath);
    const ingredients = extractIngredients(text);

    if (ingredients.length > 0) {
      console.log("Extracted Ingredients:", ingredients);

      // Generate embedding for the extracted ingredients
      const recipeEmbedding = await generateEmbeddings(
        ingredients.join(" ").toLowerCase()
      );

      // Query Chroma DB for similar recipes
      const collection = await chroma.getCollection({ name: collectionName });
      const results = await collection.query({
        queryEmbeddings: [recipeEmbedding],
        n: 5, // Get top 5 similar items
      });

      // console.log("Chroma DB Query Results:", results);

      if (results.ids.length > 0 && results.ids[0].length > 0) {
        console.log("Recommended Recipes:");
        results.ids[0].forEach((id, index) => {
          const recommendedItem = foodItems[parseInt(id)];
          console.log(
            `Top ${index + 1} Recommended Item ==> ${recommendedItem.food_name}`
          );
        });
      } else {
        console.log("No similar recipes found.");
      }
    } else {
      console.log("No ingredients found in the recipe.");
    }
  } catch (err) {
    console.error("An error occurred:", err);
  }
};

main();
