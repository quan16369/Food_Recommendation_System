const { ChromaClient } = require("chromadb");
const client = new ChromaClient();
const { HfInference } = require("@huggingface/inference");
const hf = new HfInference("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");
const foodItems = require("./FoodDataSet.js");
const collectionName = "food_collection";

async function generateEmbeddings(texts) {
  const results = await hf.featureExtraction({
    model: "sentence-transformers/all-MiniLM-L6-v2",
    inputs: texts,
  });
  return results;
}
async function classifyText(text, labels) {
  const response = await hf.request({
    model: "facebook/bart-large-mnli",
    inputs: text,
    parameters: { candidate_labels: labels },
  });
  return response;
}
async function extractFilterCriteria(query) {
  const criteria = { diet: null, cuisine: null };
  const dietLabels = [
    "vegan",
    "non-vegan",
    "vegetarian",
    "non-vegetarian",
    "pescatarian",
    "omnivore",
    "paleo",
    "ketogenic",
  ];
  const cuisineLabels = ["chinese", "indian", "japanese"];
  const dietResult = await classifyText(query, dietLabels);
  const highestDietScoreLabel = dietResult.labels[0];
  const dietScore = dietResult.scores[0];

  if (dietScore > 0.8) {
    criteria.diet = highestDietScoreLabel;
  } else {
    const cuisineResult = await classifyText(query, cuisineLabels);
    const highestCuisineScoreLabel = cuisineResult.labels[0];
    const cuisineScore = cuisineResult.scores[0];

    if (cuisineScore > 0.8) {
      criteria.cuisine = highestCuisineScoreLabel;
    }
  }
  console.log("Extracted Filter Criteria:", criteria);
  return criteria;
}
async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
  try {
    const queryEmbedding = await generateEmbeddings([queryTerm]);
    console.log(filterCriteria);
    const results = await collection.query({
      collection: collectionName,
      queryEmbeddings: queryEmbedding,
      n: 5,
    });
    if (!results || results.length === 0) {
      console.log(`No food items found similar to "${queryTerm}"`);
      return [];
    }
    let topFoodItems = results.ids[0]
      .map((id, index) => {
        return {
          id,
          score: results.distances[0][index],
          food_name: foodItems.find((item) => item.food_id.toString() === id)
            .food_name,
          food_description: foodItems.find(
            (item) => item.food_id.toString() === id
          ).food_description,
        };
      })
      .filter(Boolean);
    return topFoodItems.sort((a, b) => a.score - b.score);
  } catch (error) {
    console.error("Error during similarity search:", error);
    return [];
  }
}
async function main() {
  const query = "I want to eat Vietnamese food Which one is good for me";
  try {
    const collection = await client.getOrCreateCollection({
      name: collectionName,
    });
    const uniqueIds = new Set();
    foodItems.forEach((food, index) => {
      while (uniqueIds.has(food.food_id.toString())) {
        food.food_id = `${food.food_id}_${index}`;
      }
      uniqueIds.add(food.food_id.toString());
    });
    const foodTexts = foodItems.map(
      (food) =>
        `${food.food_name}. ${
          food.food_description
        }. Ingredients: ${food.food_ingredients.join(", ")}`
    );
    const embeddingsData = await generateEmbeddings(foodTexts);
    await collection.add({
      ids: foodItems.map((food) => food.food_id.toString()),
      documents: foodTexts,
      embeddings: embeddingsData,
    });
    const filterCriteria = await extractFilterCriteria(query);
    const initialResults = await performSimilaritySearch(
      collection,
      query,
      filterCriteria
    );
    initialResults.slice(0, 5).forEach((item, index) => {
      console.log(`Top ${index + 1} Recommended Food Name: ${item.food_name}`);
    });
  } catch (error) {
    console.error("Error:", error);
  }
}

main();
