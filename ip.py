import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
import geopandas as gpd
import folium
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('LifeExpectancyData.csv')

section = st.sidebar.radio('Which section?', ('Overview', 'Correlation', 'Predict the Future', 'Maps'))

if section == 'Overview':
    st.title('Life Expectancy')
    st.write('Since 1900 the global average life expectancy has more than doubled and is now above 70 years. The inequality of life expectancy is still very large across and within countries. In 2019 the country with the lowest life expectancy was the Central African Republic with 53 years. In Japan, life expectancy was 30 years longer. Roser, M., Ortiz-Ospina, E., & Ritchie, H. (2013)')

    st.header('Compare life expectancy across different countries from 2000 to 2015')
    selected_years = st.slider('Select years', min_value=2000, max_value=2015, value=(2000, 2015))

    countries = st.multiselect("Choose countries", df["Country"].unique(), ["Lebanon", "Canada"])
    if not countries:
        st.error("Please select at least one country.")
    else:
        data = df.loc[df["Country"].isin(countries)]
        st.write("### Life Expectancy (years)")

        # Filter data based on the selected year range
        data = data[(data['Year'] >= selected_years[0]) & (data['Year'] <= selected_years[1])]

        # Pivot the data to reshape it
        table_data = data.pivot_table(index='Country', columns='Year', values='Life Expectancy')
        table_data = table_data.reset_index()

        st.dataframe(table_data)

        # Line chart
        chart_data = pd.melt(data, id_vars=['Country', 'Year'], value_vars=['Life Expectancy'])
        chart = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x=alt.X('Year:O', title='Year', axis=alt.Axis(labels=True)),
                y=alt.Y('value:Q', title='Life Expectancy (years)'),
                color='Country:N',
                tooltip=['Country', 'Year', 'value']
            )
            .properties(width=600, height=400)
            .configure_view(strokeWidth=0)
        )
        st.altair_chart(chart, use_container_width=True)
   
 
elif section == 'Correlation':
    
    st.title('Analyze the correlation between Life Expectancy and possible factors')

    st.header('Economic Factors')
    # Filter the data for "Developing" and "Developed" status
    developing_data = df[df['Status'] == 'Developing']
    developed_data = df[df['Status'] == 'Developed']

    # Calculate the average life expectancy for each category
    avg_life_expectancy = {
        'Developing': developing_data['Life Expectancy'].mean(),
        'Developed': developed_data['Life Expectancy'].mean()
    }

    # Create a DataFrame from the average values
    avg_life_expectancy_df = pd.DataFrame(avg_life_expectancy.items(), columns=['Status', 'Average Life Expectancy'])

    # Create the bar chart
    status_chart = (
        alt.Chart(avg_life_expectancy_df)
        .mark_bar()
        .encode(
             x=alt.X('Status:N', axis=alt.Axis(labelAngle=0)),
            y='Average Life Expectancy',
            color='Status',
            tooltip=['Status', 'Average Life Expectancy']
        )
        .properties(width=300, height=200)
    )

    gdp_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='GDP',
        y='Life Expectancy',
        tooltip=['Country', 'GDP', 'Life Expectancy']
    )
    .properties(width=300, height=200)
    )

    expenditure_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Total Expenditure',
        y='Life Expectancy',
        tooltip=['Country', 'Total Expenditure', 'Life Expectancy']
    )
    .properties(width=300, height=200)
    )

    hdi_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Income Composition of Resources',
        y='Life Expectancy',
        tooltip=['Country', 'Income Composition of Resources', 'Life Expectancy']
    )
    .properties(width=300, height=200)
    )

    # Display the charts in a 2x2 grid
    st.markdown("<p style='text-align: center;'>Correlation Between Economic Factors and Life Expectancy</p>", unsafe_allow_html=True)
    st.write('The graphs below show that there is some correlation between life expectancy and the development status of the country. The correlation between Life expectancy and GDP or how much the country spends on health is less clear. The average life expectancy in deveoped countries is 79 years, while the life expectancy in developing countries is 67 years. That is a gap 12 years.')
    st.altair_chart(alt.vconcat(alt.hconcat(status_chart, gdp_chart), alt.hconcat(expenditure_chart, hdi_chart)), use_container_width=True)

    st.header('Education')
    # Create the chart
    schooling_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Schooling',
        y='Life Expectancy',
        tooltip=['Country', 'Schooling', 'Life Expectancy']).properties(width=600, height=400))

    # Display the chart in Streamlit
    st.markdown("<p style='text-align: center;'>Correlation Between Schooling and Life Expectancy</p>", unsafe_allow_html=True)
    st.write('The graph below shows that there is a clear correlation between life expectancy and the number of schooling years. The average life expectancy for people who received 15 years of schooling or more is at least 70 years')
    st.altair_chart(schooling_chart, use_container_width=True)

    st.header('Immunization')

    hepatitis_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Hepatitis B',
        y='Life Expectancy',
        tooltip=['Country', 'Hepatitis B', 'Life Expectancy']).properties(width=300, height=200))

    polio_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Polio',
        y='Life Expectancy',
        tooltip=['Country', 'Polio', 'Life Expectancy']).properties(width=300, height=200))

    diphtheria_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Diphtheria',
        y='Life Expectancy',
        tooltip=['Country', 'Diphtheria', 'Life Expectancy']).properties(width=300, height=200))

    three_vaccines_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Immunization',
        y='Life Expectancy',
        tooltip=['Country', 'Immunization', 'Life Expectancy']).properties(width=300, height=200))

    # Display the charts in a 2x2 grid
    st.markdown("<p style='text-align: center;'>Correlation Between Immunization Coverage and Life Expectancy</p>", unsafe_allow_html=True)
    st.write('The graphs below show the correlation between life expectancy and the immunization coverage for Hepatitis B, Diphtheria, and Polio. The forth chart shows the correlation between life expectancy and the average immunization coverage for all three vaccines. The data shows that there is some correlation between the immunization coverage and life expectancy, but the evidence from the visual is not conclusive.')
    st.altair_chart(alt.vconcat(alt.hconcat(hepatitis_chart, polio_chart), alt.hconcat(diphtheria_chart, three_vaccines_chart)), use_container_width=True)

    hepatitis_infant_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Hepatitis B',
        y='Infant Deaths',
        tooltip=['Country', 'Hepatitis B', 'Infant Deaths']).properties(width=300, height=200))

    polio_infant_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Polio',
        y='Infant Deaths',
        tooltip=['Country', 'Polio', 'Infant Deaths']).properties(width=300, height=200))

    diphtheria_infant_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Diphtheria',
        y='Infant Deaths',
        tooltip=['Country', 'Diphtheria', 'Infant Deaths']).properties(width=300, height=200))

    three_vaccines_infant_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Immunization',
        y='Infant Deaths',
        tooltip=['Country', 'Immunization', 'Infant Deaths']).properties(width=300, height=200))

    # Display the charts in a 2x2 grid
    st.markdown("<p style='text-align: center;'>Correlation Between Immunization Coverage and Infant Deaths</p>", unsafe_allow_html=True)
    st.write('The data also does not show a clear correlation between immunization coverage and infant deaths.')
    st.altair_chart(alt.vconcat(alt.hconcat(hepatitis_infant_chart, polio_infant_chart), alt.hconcat(diphtheria_infant_chart, three_vaccines_infant_chart)), use_container_width=True)

    st.header('Risk Factors')

    alcohol_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Alcohol',
        y='Life Expectancy',
        tooltip=['Country', 'Alcohol', 'Life Expectancy']).properties(width=300, height=200))

    measles_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
        x='Measles',
        y='Life Expectancy',
        tooltip=['Country', 'Measles', 'Life Expectancy']).properties(width=300, height=200))

    bmi_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
    x='BMI',
    y='Life Expectancy',
    tooltip=['Country', 'BMI', 'Life Expectancy']).properties(width=300, height=200))

    hiv_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
    x='HIV/AIDS',
    y='Life Expectancy',
    tooltip=['Country', 'HIV/AIDS', 'Life Expectancy']).properties(width=300, height=200))

    thinness_child_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
    x='Thinness 5-9 years',
    y='Life Expectancy',
    tooltip=['Country', 'HIV/AIDS', 'Thinness 5-9 years']).properties(width=300, height=200))

    thinness_adolescent_chart = (
    alt.Chart(df)
    .mark_circle()
    .encode(
    x='Thinness 1-19 years',
    y='Life Expectancy',
    tooltip=['Country', 'HIV/AIDS', 'Thinness 1-19 years']).properties(width=300, height=200))

    # Display the charts in a 2x2 grid
    st.markdown("<p style='text-align: center;'>Correlation Between Risk Factors and Life Expectancy</p>", unsafe_allow_html=True)
    st.write('The data shows a clear positive correlation betweeen BMI and life expectancy and some positive correlation between life expectancy and alcohol consumption. Even though a higher average BMI and more alcohol consumption are both considered risk factors on an individual level, it seems that on a country level they are associated with more developed countries which is also associated with a higher life expectancy. As for HIV/AIDS, there is a neative correlation between HIV and life expectancy, which is expected. There is a negative correlation between thinness (1-19 years old and 5-9 years old) and life expectancy. This could be explained if we consider thinness to be an indicator of malnutrition.')

    col1, col2 = st.columns(2)

    with col1:
        st.altair_chart(alcohol_chart, use_container_width=True)
        st.altair_chart(measles_chart, use_container_width=True)
        st.altair_chart(thinness_child_chart, use_container_width=True)

    with col2:
        st.altair_chart(bmi_chart, use_container_width=True)
        st.altair_chart(hiv_chart, use_container_width=True)
        st.altair_chart(thinness_adolescent_chart, use_container_width=True)        
               
    
elif section == 'Predict the Future':
    
    st.title('Using Maching Learning to Predict the Future')

    st.header('Linear Regression')
    st.write('Using a linear regression machine learning model with all variables gave the best results')
       
    # Create a copy of df called df_ml
    df_ml = df.copy()

    # Drop the 'Population' and 'GDP' columns from df_ml
    df_ml = df_ml.drop(['Population', 'GDP'], axis=1)

    # Filter the rows containing '#DIV/0!' in 'Immunization' column
    div_zero_rows = df_ml[df_ml['Immunization'] == '#DIV/0!']

    # Get the indices of the filtered rows
    div_zero_indices = div_zero_rows.index

    # Count the number of rows
    div_zero_count = len(div_zero_rows)

    # Print the count
    print("Number of rows containing '#DIV/0!':", div_zero_count)

    # Print the indices of the filtered rows
    print("Indices of rows containing '#DIV/0!':", div_zero_indices)

    # Drop the rows with stored indices
    df_ml = df_ml.drop(div_zero_indices)

    # Convert 'Immunization' column to float64 type
    df_ml['Immunization'] = df_ml['Immunization'].astype(float)

    # Split the data into training and validation sets
    train_df, val_df = train_test_split(df_ml, test_size=0.2, random_state=42)

    # Count the number of null values in train_df
    null_counts = train_df.isnull().sum()

    # Print the columns with the number of null values
    print(null_counts[null_counts > 0])

    # Split the DataFrame based on the 'Status' column
    df_train_developing = train_df[train_df['Status'] == 'Developing']
    df_train_developed = train_df[train_df['Status'] == 'Developed']

    # Split the DataFrame based on the 'Status' column
    df_val_developing = val_df[val_df['Status'] == 'Developing']
    df_val_developed = val_df[val_df['Status'] == 'Developed']

    # Initialize an empty DataFrame to store the mean values
    mean_values_train = pd.DataFrame(columns=['Column', 'Mean_Developing', 'Missing_Developing', 'Mean_Developed', 'Missing_Developed'])

    # Loop through each column in df_developing
    for column in df_train_developing.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Calculate the mean for the column in df_developing
            mean_developing_train = df_train_developing[column].mean()

            # Calculate the mean for the column in df_developed
            mean_developed_train = df_train_developed[column].mean()

            # Count the number of missing values for developing countries
            missing_developing_train = df_train_developing[column].isnull().sum()

            # Count the number of missing values for developed countries
            missing_developed_train = df_train_developed[column].isnull().sum()

            # Append the mean and missing values to the mean_values DataFrame
            mean_values_train = mean_values_train.append({'Column': column, 'Mean_Developing': mean_developing_train, 'Missing_Developing': missing_developing_train, 'Mean_Developed': mean_developed_train, 'Missing_Developed': missing_developed_train}, ignore_index=True)

    # Initialize an empty DataFrame to store the mean values
    mean_values_val = pd.DataFrame(columns=['Column', 'Mean_Developing', 'Missing_Developing', 'Mean_Developed', 'Missing_Developed'])

    # Loop through each column in df_val_developing
    for column in df_val_developing.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Calculate the mean for the column in df_developing
            mean_developing_val = df_val_developing[column].mean()

            # Calculate the mean for the column in df_developed
            mean_developed_val = df_val_developed[column].mean()

            # Count the number of missing values for developing countries
            missing_developing_val = df_val_developing[column].isnull().sum()

            # Count the number of missing values for developed countries
            missing_developed_val = df_val_developed[column].isnull().sum()

            # Append the mean and missing values to the mean_values DataFrame
            mean_values_val = mean_values_val.append({'Column': column, 'Mean_Developing': mean_developing_val, 'Missing_Developing': missing_developing_val, 'Mean_Developed': mean_developed_val, 'Missing_Developed': missing_developed_val}, ignore_index=True)


    # Loop through each column in df_train_developing
    for column in df_train_developing.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Get the mean value from mean_values_train DataFrame
            train_mean_value = mean_values_train.loc[mean_values_train['Column'] == column, 'Mean_Developing'].values[0]

            # Impute missing values in df_train_developing column with the mean value
            df_train_developing[column].fillna(train_mean_value, inplace=True)

    # Loop through each column in df_train_developed
    for column in df_train_developed.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Get the mean value from mean_values_train DataFrame
            mean_value = mean_values_train.loc[mean_values_train['Column'] == column, 'Mean_Developed'].values[0]

            # Impute missing values in df_train_developed column with the mean value
            df_train_developed[column].fillna(mean_value, inplace=True)


    # Loop through each column in df_val_developing
    for column in df_val_developing.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Get the mean value from mean_values_val DataFrame
            val_mean_value = mean_values_val.loc[mean_values_val['Column'] == column, 'Mean_Developing'].values[0]

            # Impute missing values in df_val_developing column with the mean value
            df_val_developing[column].fillna(val_mean_value, inplace=True)

    # Loop through each column in df_val_developed
    for column in df_val_developed.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Get the mean value from mean_values_val DataFrame
            mean_value = mean_values_val.loc[mean_values_val['Column'] == column, 'Mean_Developed'].values[0]

            # Impute missing values in df_val_developed column with the mean value
            df_val_developed[column].fillna(mean_value, inplace=True)

    # Initialize an empty DataFrame to store the missing values count
    missing_values = pd.DataFrame(columns=['Column', 'Missing_Developing', 'Missing_Developed'])

    # Loop through each column in df_train_developing
    for column in df_train_developing.columns:
        # Exclude the 'Country', 'Year', and 'Status' columns
        if column not in ['Country', 'Year', 'Status']:
            # Count the number of missing values for developing countries
            missing_developing = df_train_developing[column].isnull().sum()

            # Count the number of missing values for developed countries
            missing_developed = df_train_developed[column].isnull().sum()

            # Append the column name and missing values count to the missing_values DataFrame
            missing_values = missing_values.append({'Column': column, 'Missing_Developing': missing_developing, 'Missing_Developed': missing_developed}, ignore_index=True)


    # Concatenate df_developing and df_developed DataFrames
    train_df_clean = pd.concat([df_train_developing, df_train_developed], ignore_index=True)

    # Concatenate df_developing and df_developed DataFrames
    val_df_clean = pd.concat([df_val_developing, df_val_developed], ignore_index=True)

    # check if the concatinated data frame has any missing values
    train_df_clean.isnull().any()

    # check if the concatinated data frame has any missing values
    val_df_clean.isnull().any()

    # Create a new column 'Status_Encoded' based on 'Status'
    train_df_clean['Status_Encoded'] = train_df_clean['Status'].map({'Developing': 1, 'Developed': 2})
    val_df_clean['Status_Encoded'] = val_df_clean['Status'].map({'Developing': 1, 'Developed': 2})

    # Exclude the 'Country', 'Year', and 'Status' columns from the training data
    train_cols = [col for col in train_df_clean.columns if col not in ['Country', 'Year', 'Status', 'Life Expectancy']]

    # Extract the features (X) and target variable (y) from train_df_clean
    X_train = train_df_clean[train_cols]
    y_train = train_df_clean['Life Expectancy']

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Perform prediction on val_df_clean
    X_val = val_df_clean[train_cols]
    y_pred = model.predict(X_val)

    # Calculate the mean squared error on val_df_clean
    mse = mean_squared_error(val_df_clean['Life Expectancy'], y_pred)

    # Display the mean squared error on Streamlit
    st.header('Accuracy Metric')
    st.write("Mean Squared Error:", mse)

    # Create a DataFrame for coefficients
    coefficients = pd.DataFrame({'Variable': X_train.columns, 'Coefficient': model.coef_})

    # Display the coefficients on Streamlit
    st.header('Model Coefficients')
    st.write("Coefficients:")
    st.write(coefficients)

    # Show top 5 predictors based on coefficients
    top_predictors = coefficients.nlargest(5, 'Coefficient')

    # Display the top 5 predictors on Streamlit
    st.header('The Top Five Predictors')
    st.write("The top 5 Predictors are:")
    st.write(top_predictors)


elif section == 'Maps':
    # Calculate the mean life expectancy for each country
    mean_life_expectancy = df.groupby('Country')['Life Expectancy'].mean()

    # Create a new column 'life_expectancy_mean' with the mean values
    df['life_expectancy_mean'] = df['Country'].map(mean_life_expectancy)

    # Create a new column 'life_expectancy_length' based on 'life_expectancy_mean'
    df['life_expectancy_length'] = pd.cut(df['life_expectancy_mean'],
                                          bins=[0, 63, 75, float('inf')],
                                          labels=['short', 'average', 'long'])

    # Group by 'life_expectancy_length' and retrieve unique country names
    short_life_expectancy = df[df['life_expectancy_length'] == 'short'].groupby('Country').groups.keys()
    average_life_expectancy = df[df['life_expectancy_length'] == 'average'].groupby('Country').groups.keys()
    long_life_expectancy = df[df['life_expectancy_length'] == 'long'].groupby('Country').groups.keys()

    # Convert the keys to lists
    short_life_expectancy = list(short_life_expectancy)
    average_life_expectancy = list(average_life_expectancy)
    long_life_expectancy = list(long_life_expectancy)

    # Read the GeoJSON file
    geojson_file = '/Users/hassancoudsi/Documents/AUB/MSBA/350HealthcareAnalytics/IndividualProject/world-administrative-boundaries.geojson'
    gdf = gpd.read_file(geojson_file)
    
    # Create a map using Folium
    try:
        m = folium.Map()
        
        # Get unique country names from the lists
        all_countries = set(short_life_expectancy + average_life_expectancy + long_life_expectancy)

        # Color the countries
        for feature in gdf.iterrows():
            country_name = feature[1]['name']
            if country_name in all_countries:
                if country_name in short_life_expectancy:
                    color = 'red'  # Assign a color for short life expectancy countries
                elif country_name in average_life_expectancy:
                    color = 'orange'  # Assign a color for average life expectancy countries
                elif country_name in long_life_expectancy:
                    color = 'green'  # Assign a color for long life expectancy countries
                else:
                    color = 'blue'  # Assign a color for other countries in the dataset
            else:
                color = 'gray'  # Assign a color for countries not in the dataset

            geometry = feature[1].geometry
            if not geometry.is_empty:
                folium.GeoJson(
                    geometry.__geo_interface__,
                    name=country_name,
                    style_function=lambda feature, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    }
                ).add_to(m)

        # Create an elegant legend
        legend_html = '''
        <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 120px; height: 120px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color: white;
                     ">
            <p style="margin: 0 5px; text-align:center;">Legend</p>
            <hr style="margin: 5px;">
            <table style="margin: 5px;">
              <tr>
                <td><span style='display:inline-block;width:10px;height:10px;background:red;margin-right:5px;'></span>Below 63</td>
              </tr>
              <tr>
                <td><span style='display:inline-block;width:10px;height:10px;background:orange;margin-right:5px;'></span>63 to 75</td>
              </tr>
              <tr>
                <td><span style='display:inline-block;width:10px;height:10px;background:green;margin-right:5px;'></span>Above 75</td>
              </tr>
            </table>
        </div>
        '''

        # Add the legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))

        # Display the map on Streamlit
        st.write(m)

    except KeyError:
        pass

