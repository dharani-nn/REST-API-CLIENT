import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;

import org.json.JSONObject;

public class WeatherClient {

    private static final String API_KEY = "YOUR_API_KEY";  // Replace with your OpenWeatherMap API key
    private static final String BASE_URL = "https://api.openweathermap.org/data/2.5/weather?q=";

    public static void main(String[] args) {
        String city = "London";
        String urlString = BASE_URL + city + "&appid=" + API_KEY + "&units=metric";

        try {
            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            int responseCode = conn.getResponseCode();
            if (responseCode == HttpURLConnection.HTTP_OK) {
                BufferedReader in = new BufferedReader(
                        new InputStreamReader(conn.getInputStream()));
                String inputLine;
                StringBuilder response = new StringBuilder();

                while ((inputLine = in.readLine()) != null) {
                    response.append(inputLine);
                }
                in.close();

                // Parse JSON response
                JSONObject json = new JSONObject(response.toString());
                String cityName = json.getString("name");
                JSONObject main = json.getJSONObject("main");
                double temp = main.getDouble("temp");
                double feelsLike = main.getDouble("feels_like");
                int humidity = main.getInt("humidity");

                JSONObject weather = json.getJSONArray("weather").getJSONObject(0);
                String description = weather.getString("description");

                System.out.println("Weather in " + cityName + ":");
                System.out.println("Temperature: " + temp + "°C");
                System.out.println("Feels Like: " + feelsLike + "°C");
                System.out.println("Humidity: " + humidity + "%");
                System.out.println("Condition: " + description);
            } else {
                System.out.println("GET request failed. Response Code: " + responseCode);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}