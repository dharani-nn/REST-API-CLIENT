package com.example;

import org.apache.mahout.cf.taste.common.TasteException;
import org.apache.mahout.cf.taste.impl.model.file.FileDataModel;
import org.apache.mahout.cf.taste.impl.neighborhood.NearestNUserNeighborhood;
import org.apache.mahout.cf.taste.impl.recommender.GenericUserBasedRecommender;
import org.apache.mahout.cf.taste.impl.similarity.PearsonCorrelationSimilarity;
import org.apache.mahout.cf.taste.model.DataModel;
import org.apache.mahout.cf.taste.model.Preference;
import org.apache.mahout.cf.taste.model.PreferenceArray;
import org.apache.mahout.cf.taste.neighborhood.UserNeighborhood;
import org.apache.mahout.cf.taste.recommender.RecommendedItem;
import org.apache.mahout.cf.taste.recommender.Recommender;
import org.apache.mahout.cf.taste.similarity.UserSimilarity;
import org.apache.mahout.cf.taste.impl.common.LongPrimitiveIterator;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class RecommenderSystem {

    public static void main(String[] args) {
        try {
            // Load dataset from CSV file - ensure this path is correct
            DataModel model = new FileDataModel(new File("data/dataset.csv"));
            System.out.println("Dataset loaded.");

            // Create similarity measure
            UserSimilarity similarity = new PearsonCorrelationSimilarity(model);

            // Define neighborhood - 2 nearest neighbors
            UserNeighborhood neighborhood = new NearestNUserNeighborhood(2, similarity, model);

            // Build recommender
            Recommender recommender = new GenericUserBasedRecommender(model, neighborhood, similarity);

            // Iterate all users and print their preferences and neighbors' preferences, then recommendations
            LongPrimitiveIterator users = model.getUserIDs();
            while (users.hasNext()) {
                long userId = users.nextLong();
                System.out.println("User " + userId + " preferences:");
                PreferenceArray userPrefs = model.getPreferencesFromUser(userId);
                for (int i = 0; i < userPrefs.length(); i++) {
                    Preference pref = userPrefs.get(i);
                    System.out.printf("  Item %d: %.2f%n", pref.getItemID(), pref.getValue());
                }

                // Show neighbors
                long[] neighborIds = neighborhood.getUserNeighborhood(userId);
                System.out.println("User " + userId + " neighbors:");
                for (long nId : neighborIds) {
                    System.out.println("  Neighbor " + nId + " preferences:");
                    PreferenceArray prefs = model.getPreferencesFromUser(nId);
                    for (int i = 0; i < prefs.length(); i++) {
                        Preference pref = prefs.get(i);
                        System.out.printf("    Item %d: %.2f%n", pref.getItemID(), pref.getValue());
                    }
                }

                // Get and print recommendations
                List<RecommendedItem> recommendations = recommender.recommend(userId, 3);
                System.out.println("User " + userId + " recommendations:");
                if (recommendations.isEmpty()) {
                    System.out.println("  (No recommendations)");
                } else {
                    for (RecommendedItem recommendation : recommendations) {
                        System.out.printf("  Item %d with score %.2f%n", recommendation.getItemID(), recommendation.getValue());
                    }
                }
                System.out.println(); // blank line for readability between users
            }

        } catch (IOException | TasteException e) {
            e.printStackTrace();
        }
    }
}
